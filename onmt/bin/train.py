#!/usr/bin/env python
"""Train models with dynamic data."""
import sys
import torch
from functools import partial

# import onmt.opts as opts
from onmt.utils.distributed import ErrorHandler, consumer, batch_producer
from onmt.utils.misc import set_random_seed
from onmt.modules.embeddings import prepare_pretrained_embeddings
from onmt.utils.logging import init_logger, logger

from onmt.models.model_saver import load_checkpoint
from onmt.train_single import main as single_main, async_main, _get_model_opts, _build_train_iter

from onmt.utils.parse import ArgumentParser
from onmt.opts import train_opts
from onmt.inputters.corpus import save_transformed_sample
from onmt.inputters.fields import build_dynamic_fields, save_fields, \
    load_fields
from onmt.transforms import make_transforms, save_transforms, \
    get_specials, get_transforms_cls
from onmt.model_builder import build_model
from onmt.utils.optimizers import Optimizer
from onmt.constants import ModelTask, TrainMode
from onmt.modules.rewards import UnsuperReward

# Set sharing strategy manually instead of default based on the OS.
torch.multiprocessing.set_sharing_strategy('file_system')


def prepare_fields_transforms(opt):
    """Prepare or dump fields & transforms before training."""
    transforms_cls = get_transforms_cls(opt._all_transform)
    specials = get_specials(opt, transforms_cls)

    fields = build_dynamic_fields(
        opt, src_specials=specials['src'], tgt_specials=specials['tgt'])

    # maybe prepare pretrained embeddings, if any
    prepare_pretrained_embeddings(opt, fields)

    if opt.dump_fields:
        save_fields(fields, opt.save_data, overwrite=opt.overwrite)
    if opt.dump_transforms or opt.n_sample != 0:
        transforms = make_transforms(opt, transforms_cls, fields)
    if opt.dump_transforms:
        save_transforms(transforms, opt.save_data, overwrite=opt.overwrite)
    if opt.n_sample != 0:
        logger.warning(
            "`-n_sample` != 0: Training will not be started. "
            f"Stop after saving {opt.n_sample} samples/corpus.")
        save_transformed_sample(opt, transforms, n_sample=opt.n_sample)
        logger.info(
            "Sample saved, please check it before restart training.")
        sys.exit()
    return fields, transforms_cls


def _init_train(opt):
    """Common initilization stuff for all training process."""
    ArgumentParser.validate_prepare_opts(opt)

    if opt.train_from:
        # Load checkpoint if we resume from a previous training.
        checkpoint = load_checkpoint(ckpt_path=opt.train_from)
        fields = load_fields(opt.save_data, checkpoint)
        transforms_cls = get_transforms_cls(opt._all_transform)
        if (hasattr(checkpoint["opt"], '_all_transform') and
                len(opt._all_transform.symmetric_difference(
                    checkpoint["opt"]._all_transform)) != 0):
            _msg = "configured transforms is different from checkpoint:"
            new_transf = opt._all_transform.difference(
                checkpoint["opt"]._all_transform)
            old_transf = checkpoint["opt"]._all_transform.difference(
                opt._all_transform)
            if len(new_transf) != 0:
                _msg += f" +{new_transf}"
            if len(old_transf) != 0:
                _msg += f" -{old_transf}."
            logger.warning(_msg)
        if opt.update_vocab:
            logger.info("Updating checkpoint vocabulary with new vocabulary")
            fields, transforms_cls = prepare_fields_transforms(opt)
    else:
        checkpoint = None
        fields, transforms_cls = prepare_fields_transforms(opt)

    # Report src and tgt vocab sizes
    for side in ['src', 'tgt']:
        f = fields[side]
        try:
            f_iter = iter(f)
        except TypeError:
            f_iter = [(side, f)]
        for sn, sf in f_iter:
            if sf.use_vocab:
                logger.info(' * %s vocab size = %d' % (sn, len(sf.vocab)))
    return checkpoint, fields, transforms_cls


def train(opt):
    init_logger(opt.log_file)
    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)

    set_random_seed(opt.seed, False)

    checkpoint, fields, transforms_cls = _init_train(opt)

    if checkpoint is not None and opt.async and opt.model_task == ModelTask.AC:
        model_params = checkpoint['model']
        gen_params = checkpoint['generator']

        checkpoint = None
        opt.train_from = ''

    if opt.async or opt.model_task == ModelTask.A3C and opt.train_mode in [TrainMode.CRITIC, TrainMode.AC]:

        print('Performing Async Training')

        model_opt = _get_model_opts(opt, checkpoint=checkpoint)

        # if len(opt.gpu_ranks) > 2:
        global_gpu_id = opt.gpu_ranks[-1] + 1

        print('Building the global model')
        # Build a global model.
        # global_model = build_model(opt, opt, fields, checkpoint, gpu_id=global_gpu_id)
        global_model = build_model(model_opt, opt, fields, checkpoint, gpu_id=global_gpu_id)

        if opt.async and opt.model_task == ModelTask.AC:
            global_model.load_state_dict(model_params, strict=False)
            global_model.generator.load_state_dict(gen_params, strict=False)

        global_model.share_memory()

        # Build optimizer.
        if opt.train_from and checkpoint is not None:
            actor_optim = Optimizer.from_opt(global_model.actor, opt, checkpoint=checkpoint, ac_optim_opt='actor')
            critic_optim = Optimizer.from_opt(global_model.critic, opt, checkpoint=checkpoint, ac_optim_opt='critic')
            optim = (actor_optim, critic_optim)
        else:
            actor_optim = Optimizer.from_opt(global_model.actor, opt, checkpoint=checkpoint, ac_optim_opt='actor')
            critic_optim = Optimizer.from_opt(global_model.critic, opt, checkpoint=checkpoint, ac_optim_opt='critic')
            optim = (actor_optim, critic_optim)

        # if opt.model_task == ModelTask.AC:
        #     actor_optim = Optimizer.from_opt(global_model.actor, opt, checkpoint=None, ac_optim_opt='actor')
        #     critic_optim = Optimizer.from_opt(global_model.critic, opt, checkpoint=None, ac_optim_opt='critic')
        #     optim = (actor_optim, critic_optim)

        # unsuper_reward = UnsuperReward(fields, opt.w_fluency, opt.w_tlss, opt.w_slss, global_gpu_id, opt.norm_unsuper_reward)
        # unsuper_reward = None

        train_process = partial(
            async_main,
            global_model=global_model,
            optim=optim,
            # global_gpu_id=global_gpu_id,
            # unsuper_reward=unsuper_reward,
            model_opt=opt,
            fields=fields,
            transforms_cls=transforms_cls,
            checkpoint=checkpoint)
    else:
        print('Performing Sync Training')

        train_process = partial(
            single_main,
            fields=fields,
            transforms_cls=transforms_cls,
            checkpoint=checkpoint)

    nb_gpu = len(opt.gpu_ranks)

    print('Number of gpus: {}'.format(nb_gpu))
    print('GPU RANKS: {}'.format(opt.gpu_ranks))

    if opt.world_size > 1:

        queues = []
        mp = torch.multiprocessing.get_context('spawn')

        print('Creating semaphore')

        print('World size: {}'.format(opt.world_size))

        semaphore = mp.Semaphore(opt.world_size * opt.queue_size)
        # Create a thread to listen for errors in the child processes.
        error_queue = mp.SimpleQueue()
        error_handler = ErrorHandler(error_queue)
        # Train with multiprocessing.
        procs = []

        print('Creating consumers')

        for device_id in range(nb_gpu):

            print('Constructing mp queue')

            q = mp.Queue(opt.queue_size)
            queues += [q]

            print('Deploying consumer')

            procs.append(mp.Process(target=consumer, args=(
                train_process, opt, device_id, error_queue, q, semaphore),
                daemon=True))

            print('procs: {}'.format(procs))

            print('Starting consumer')

            procs[device_id].start()
            logger.info(" Starting process pid: %d  " % procs[device_id].pid)
            error_handler.add_child(procs[device_id].pid)
        producers = []

        print('Creating producers')
        # This does not work if we merge with the first loop, not sure why
        for device_id in range(nb_gpu):
            # Get the iterator to generate from

            print('Building train iter')

            train_iter = _build_train_iter(
                opt, fields, transforms_cls, stride=nb_gpu, offset=device_id)

            print('Deploying producer')

            producer = mp.Process(target=batch_producer,
                                  args=(train_iter, queues[device_id],
                                        semaphore, opt, device_id),
                                  daemon=True)
            producers.append(producer)
            producers[device_id].start()
            logger.info(" Starting producer process pid: {}  ".format(
                producers[device_id].pid))
            error_handler.add_child(producers[device_id].pid)

        for p in procs:
            p.join()
        # Once training is done, we can terminate the producers
        for p in producers:
            p.terminate()

    elif nb_gpu == 1:  # case 1 GPU only
        train_process(opt, device_id=opt.device_id)
    else:   # case only CPU
        train_process(opt, device_id=-1)


def _get_parser():
    parser = ArgumentParser(description='train.py')
    train_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt, unknown = parser.parse_known_args()
    train(opt)


if __name__ == "__main__":
    main()
