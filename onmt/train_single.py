#!/usr/bin/env python
"""Training on a single process."""
import torch

from onmt.inputters.inputter import IterOnDevice
from onmt.model_builder import build_model
from onmt.utils.optimizers import Optimizer
from onmt.utils.misc import set_random_seed
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser

from onmt.inputters.dynamic_iterator import build_dynamic_dataset_iter
from onmt.constants import ModelTask, TrainMode
from onmt.modules.rewards import UnsuperReward

def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)


def _get_model_opts(opt, checkpoint=None):
    """Get `model_opt` to build model, may load from `checkpoint` if any."""
    if checkpoint is not None:
        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        if (opt.tensorboard_log_dir == model_opt.tensorboard_log_dir and
                hasattr(model_opt, 'tensorboard_log_dir_dated')):
            # ensure tensorboard output is written in the directory
            # of previous checkpoints
            opt.tensorboard_log_dir_dated = model_opt.tensorboard_log_dir_dated
        # Override checkpoint's update_embeddings as it defaults to false
        model_opt.update_vocab = opt.update_vocab
        # Override model task in casing of using pre-trained models from other tasks
        model_opt.model_task = opt.model_task
        # Override train mode during the current configuration
        model_opt.train_mode = opt.train_mode
    else:
        model_opt = opt
    return model_opt


def _build_valid_iter(opt, fields, transforms_cls):
    """Build iterator used for validation."""
    valid_iter = build_dynamic_dataset_iter(
        fields, transforms_cls, opt, is_train=False)
    return valid_iter


def _build_train_iter(opt, fields, transforms_cls, stride=1, offset=0):
    """Build training iterator."""
    train_iter = build_dynamic_dataset_iter(
        fields, transforms_cls, opt, is_train=True,
        stride=stride, offset=offset)
    return train_iter


def main(opt, fields, transforms_cls, checkpoint, device_id,
         batch_queue=None, semaphore=None):
    """Start training on `device_id`."""
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    configure_process(opt, device_id)
    init_logger(opt.log_file)

    model_opt = _get_model_opts(opt, checkpoint=checkpoint)

    print('Building model')
    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    model.count_parameters(log=logger.info)

    # Build optimizer.
    if model_opt.model_task not in [ModelTask.AC, ModelTask.A2C, ModelTask.A3C, ModelTask.PPO]:
        optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)
    elif opt.train_from and checkpoint is not None:
        actor_optim = Optimizer.from_opt(model.actor, opt, checkpoint=checkpoint, ac_optim_opt='actor')
        critic_optim = Optimizer.from_opt(model.critic, opt, checkpoint=checkpoint, ac_optim_opt='critic')
        optim = (actor_optim, critic_optim)
    else:
        actor_optim = Optimizer.from_opt(model.actor, opt, checkpoint=checkpoint)
        critic_optim = Optimizer.from_opt(model.critic, opt, checkpoint=checkpoint)
        optim = (actor_optim, critic_optim)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    print('Using Unsuper Reward: {}'.format(opt.unsuper_reward))

    unsuper_reward = None
    if opt.unsuper_reward and (opt.train_mode != TrainMode.ACTOR or model_opt.model_task == ModelTask.ACSE):
        unsuper_reward = UnsuperReward(fields, opt.w_fluency, opt.w_tlss,
                                       opt.w_slss, device_id, opt.norm_unsuper_reward)

    trainer = build_trainer(
        opt, device_id, model, fields,
        optim, model_saver=model_saver,
        unsuper_reward=unsuper_reward)

    if batch_queue is None:
        _train_iter = _build_train_iter(opt, fields, transforms_cls)
        train_iter = IterOnDevice(_train_iter, device_id)
    else:
        assert semaphore is not None, \
            "Using batch_queue requires semaphore as well"

        def _train_iter():
            while True:
                batch = batch_queue.get()
                semaphore.release()
                # Move batch to specified device
                IterOnDevice.batch_to_device(batch, device_id)
                yield batch

        train_iter = _train_iter()

    valid_iter = _build_valid_iter(opt, fields, transforms_cls)
    if valid_iter is not None:
        valid_iter = IterOnDevice(valid_iter, device_id)

    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0

    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=opt.valid_steps)

    if trainer.report_manager.tensorboard_writer is not None:
        trainer.report_manager.tensorboard_writer.close()

def async_main(global_model, optim, model_opt, opt, fields, transforms_cls, checkpoint, device_id,
         batch_queue=None, semaphore=None):
    """Start training on `device_id`."""
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    configure_process(opt, device_id)
    init_logger(opt.log_file)

    print('Building the local model | device_id: {}'.format(device_id))
    # Build a local A2C model.
    model = build_model(model_opt, opt, fields, checkpoint)
    model.count_parameters(log=logger.info)
    model.actor.load_state_dict(global_model.actor.state_dict())
    model.critic.load_state_dict(global_model.critic.state_dict())
    model.generator.load_state_dict(global_model.generator.state_dict())

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, global_model, fields, optim)

    unsuper_reward = UnsuperReward(fields, opt.w_fluency, opt.w_tlss, opt.w_slss, device_id, opt.norm_unsuper_reward)

    trainer = build_trainer(
        opt, device_id, model, fields, optim,
        model_saver=model_saver, global_model=global_model,
        unsuper_reward=unsuper_reward)

    if batch_queue is None:
        _train_iter = _build_train_iter(opt, fields, transforms_cls)
        train_iter = IterOnDevice(_train_iter, device_id)
    else:
        assert semaphore is not None, \
            "Using batch_queue requires semaphore as well"

        def _train_iter():
            while True:
                batch = batch_queue.get()
                semaphore.release()
                # Move batch to specified device
                IterOnDevice.batch_to_device(batch, device_id)
                yield batch

        train_iter = _train_iter()

    valid_iter = _build_valid_iter(opt, fields, transforms_cls)
    if valid_iter is not None:
        valid_iter = IterOnDevice(valid_iter, device_id)

    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0

    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=opt.valid_steps)

    if trainer.report_manager.tensorboard_writer is not None:
        trainer.report_manager.tensorboard_writer.close()