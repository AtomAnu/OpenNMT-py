""" Onmt NMT Model base class definition """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from onmt.constants import TrainMode, PolicyStrategy
import numpy as np
# from onmt.translate.greedy_search import sample_with_temperature

class BaseModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder / decoder or decoder only model.
    """

    def __init__(self, encoder, decoder):
        super(BaseModel, self).__init__()

    def forward(self, src, tgt, lengths, bptt=False, with_align=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        raise NotImplementedError

    def update_dropout(self, dropout):
        raise NotImplementedError

    def count_parameters(self, log=print):
        raise NotImplementedError


class NMTModel(BaseModel):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.
    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder

        # TODO to be removed
        self.max_len = 0

    def forward(self, src, tgt, lengths, bptt=False, with_align=False):
        dec_in = tgt[:-1]  # exclude last target from inputs

        if self.max_len < dec_in.shape[0]:
            self.max_len = dec_in.shape[0]

        enc_state, memory_bank, lengths = self.encoder(src, lengths)

        if not bptt:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(dec_in, memory_bank,
                                      memory_lengths=lengths,
                                      with_align=with_align)

        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

    def count_parameters(self, log=print):
        """Count number of parameters in model (& print with `log` callback).

        Returns:
            (int, int):
            * encoder side parameter count
            * decoder side parameter count
        """

        enc, dec = 0, 0
        for name, param in self.named_parameters():
            if 'encoder' in name:
                enc += param.nelement()
            else:
                dec += param.nelement()
        if callable(log):
            log('encoder: {}'.format(enc))
            log('decoder: {}'.format(dec))
            log('* number of parameters: {}'.format(enc + dec))
        return enc, dec

class Actor(nn.Module):

    def __init__(self, encoder, decoder):
        super(Actor, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False, with_align=False, train_mode=TrainMode.ACTOR, tgt_field=None,
                policy_strategy=PolicyStrategy.Categorical, policy_topk_sampling=-1, policy_sampling_temperature=1,
                policy_topp_sampling=-1):

        if train_mode == TrainMode.ACTOR:
            enc_state, memory_bank, lengths = self.encoder(src, lengths)

            if not bptt:
                self.decoder.init_state(src, memory_bank, enc_state)

            dec_in = tgt[:-1]  # exclude last target from inputs
            dec_out, attns = self.decoder(dec_in, memory_bank,
                                          memory_lengths=lengths,
                                          with_align=with_align)

            return dec_out, attns
        elif train_mode == TrainMode.CRITIC:
            with torch.no_grad():
                return self._step_wise_forward(src, tgt, lengths, bptt, with_align,
                                               tgt_field, policy_strategy,
                                               policy_topk_sampling, policy_sampling_temperature,
                                               policy_topp_sampling)
        else:
            with torch.enable_grad():
                return self._step_wise_forward(src, tgt, lengths, bptt, with_align,
                                               tgt_field, policy_strategy,
                                               policy_topk_sampling, policy_sampling_temperature,
                                               policy_topp_sampling)

    def _step_wise_forward(self, src, tgt, lengths, bptt=False, with_align=False, tgt_field=None,
                           policy_strategy=PolicyStrategy.Categorical, policy_topk_sampling=-1,
                           policy_sampling_temperature=1, policy_topp_sampling=-1):

        enc_state, memory_bank, lengths = self.encoder(src, lengths)

        if not bptt:
            self.decoder.init_state(src, memory_bank, enc_state)

        gen_seq = tgt[0].unsqueeze(0)
        gen_tok = gen_seq

        # TODO make gen_seq sequence length more flexible
        for step in range(0, tgt.shape[0]):
            dec_out, attns = self.decoder(gen_tok, memory_bank,
                                          step=step,
                                          memory_lengths=lengths,
                                          with_align=with_align)

            logits = self.generator[0:2](dec_out.squeeze(0))
            gen_tok = self._choose_tok(logits, policy_strategy,
                                       policy_topk_sampling, policy_sampling_temperature, policy_topp_sampling)
            gen_seq = torch.cat([gen_seq, gen_tok], dim=0)

            scores = self.generator[2](logits).unsqueeze(0)

            if step == 0:
                log_policy_dist = scores
            else:
                log_policy_dist = torch.cat([log_policy_dist, scores], dim=0)

        gen_seq_mask, policy_mask = self._compute_output_mask(gen_seq, tgt_field.base_field.vocab.stoi[tgt_field.base_field.eos_token])
        gen_seq = gen_seq * gen_seq_mask.to(torch.int64) + (~gen_seq_mask).to(torch.int64)
        log_policy_dist = log_policy_dist * policy_mask.to(torch.int64) + (~policy_mask).to(torch.int64)

        return gen_seq, log_policy_dist

    def _choose_tok(self, logits, policy_strategy=PolicyStrategy.Categorical,
                    policy_topk_sampling = -1, policy_sampling_temperature = 1, policy_topp_sampling = -1):

        if policy_strategy == PolicyStrategy.Greedy:
            gen_tok = self.sample_with_temperature(logits, sampling_temp=1, keep_topk=10, keep_topp=-1)[0]
        elif policy_strategy == PolicyStrategy.Categorical:
            gen_tok = self.sample_with_temperature(logits, sampling_temp=policy_sampling_temperature,
                                              keep_topk=policy_topk_sampling, keep_topp=policy_topp_sampling)[0]
        return gen_tok.unsqueeze(0).to('cuda')

    def sample_topp(self, logits, keep_topp):
        sorted_logits, sorted_indices = torch.sort(logits,
                                                   descending=True,
                                                   dim=1)

        cumulative_probs = torch.cumsum(F.softmax(sorted_logits,
                                                  dim=-1), dim=-1)
        sorted_indices_to_keep = cumulative_probs.lt(keep_topp)

        # keep indices until overflowing p
        cumsum_mask = sorted_indices_to_keep.cumsum(dim=1)
        last_included = cumsum_mask[:, -1:]
        last_included.clamp_(0, sorted_indices_to_keep.size()[1] - 1)
        sorted_indices_to_keep = sorted_indices_to_keep.scatter_(
            1, last_included, 1)

        # Set all logits that are not in the top-p to -10000.
        # This puts the probabilities close to 0.
        keep_indices = sorted_indices_to_keep.scatter(
            1,
            sorted_indices,
            sorted_indices_to_keep,
        )
        return logits.masked_fill(~keep_indices, -10000)

    def sample_topk(self, logits, keep_topk):
        top_values, _ = torch.topk(logits, keep_topk, dim=1)
        kth_best = top_values[:, -1].view([-1, 1])
        kth_best = kth_best.repeat([1, logits.shape[1]]).float()

        # Set all logits that are not in the top-k to -10000.
        # This puts the probabilities close to 0.
        ignore = torch.lt(logits, kth_best)
        return logits.masked_fill(ignore, -10000)

    def sample_with_temperature(self, logits, sampling_temp, keep_topk, keep_topp):
        """Select next tokens randomly from the top k possible next tokens.

        Samples from a categorical distribution over the ``keep_topk`` words using
        the category probabilities ``logits / sampling_temp``.

        Args:
            logits (FloatTensor): Shaped ``(batch_size, vocab_size)``.
                These can be logits (``(-inf, inf)``) or log-probs (``(-inf, 0]``).
                (The distribution actually uses the log-probabilities
                ``logits - logits.logsumexp(-1)``, which equals the logits if
                they are log-probabilities summing to 1.)
            sampling_temp (float): Used to scale down logits. The higher the
                value, the more likely it is that a non-max word will be
                sampled.
            keep_topk (int): This many words could potentially be chosen. The
                other logits are set to have probability 0.
            keep_topp (float): Keep most likely words until the cumulated
                probability is greater than p. If used with keep_topk: both
                conditions will be applied

        Returns:
            (LongTensor, FloatTensor):

            * topk_ids: Shaped ``(batch_size, 1)``. These are
              the sampled word indices in the output vocab.
            * topk_scores: Shaped ``(batch_size, 1)``. These
              are essentially ``(logits / sampling_temp)[topk_ids]``.
        """

        if sampling_temp == 0.0 or keep_topk == 1:
            # For temp=0.0, take the argmax to avoid divide-by-zero errors.
            # keep_topk=1 is also equivalent to argmax.
            topk_scores, topk_ids = logits.topk(1, dim=-1)
            if sampling_temp > 0:
                topk_scores /= sampling_temp
        else:
            logits = torch.div(logits, sampling_temp)
            if keep_topp > 0:
                logits = self.sample_topp(logits, keep_topp)
            if keep_topk > 0:
                logits = self.sample_topk(logits, keep_topk)
            dist = torch.distributions.Multinomial(
                logits=logits, total_count=1)
            topk_ids = torch.argmax(dist.sample(), dim=1, keepdim=True)
            topk_scores = logits.gather(dim=1, index=topk_ids)
        return topk_ids, topk_scores

    def _categorical_sampling(self, scores):

        gen_tok = []
        for logits in scores[0]:

            dist = Categorical(logits=logits)
            selected_tok = dist.sample().cpu().numpy().item()
            gen_tok.append(selected_tok)

        return torch.tensor(gen_tok).view(1,-1,1).to('cuda')

    def _generate_epsilon_greedy_policy(self, best_tok, vocab_size, epsilon):

        greedy_action_prob = 1 - epsilon + (epsilon / vocab_size)
        non_greedy_action_prob = epsilon / vocab_size

        policy = np.zeros(vocab_size)
        policy += non_greedy_action_prob
        policy[best_tok] = greedy_action_prob

        return policy

    def _compute_output_mask(self, gen_seq, eos_token):
        # to be used when the decoder conditions on its own output
        eos_idx = gen_seq.eq(eos_token).to(torch.int64)
        value_range = torch.arange(gen_seq.shape[0], 0, -1).to('cuda')
        output_multiplied = (eos_idx.transpose(0, 2) * value_range).transpose(0, 2)
        first_eos_idx = torch.argmax(output_multiplied, 0, keepdim=True).view(-1)

        gen_seq_mask = torch.ones(gen_seq.shape[0], gen_seq.shape[1], gen_seq.shape[2]).to('cuda')
        policy_mask = torch.ones(gen_seq.shape[0]-1, gen_seq.shape[1], gen_seq.shape[2]).to('cuda')

        for row in range(0, gen_seq.shape[1]):
            gen_seq_mask[first_eos_idx[row] + 1:, row] = 0
            policy_mask[first_eos_idx[row]:, row] = 0

        return gen_seq_mask.to(torch.bool), policy_mask.to(torch.bool)

    def update_dropout(self, dropout):

        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

    def count_parameters(self, log=print):
        """Count number of parameters in model (& print with `log` callback).

        Returns:
            (int, int):
            * encoder side parameter count
            * decoder side parameter count
        """

        enc, dec = 0, 0
        for name, param in self.named_parameters():
            if 'encoder' in name:
                enc += param.nelement()
            else:
                dec += param.nelement()
        if callable(log):
            log('actor encoder: {}'.format(enc))
            log('actor decoder: {}'.format(dec))

        return enc, dec

class CriticQ(nn.Module):

    def __init__(self, encoder, decoder, output_layer):
        super(CriticQ, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_layer = output_layer

    def forward(self, tgt, gen_seq, eos_token, lengths=None, bptt=False, with_align=False):

        output_mask = self._compute_output_mask(gen_seq, eos_token)

        lengths = torch.tensor([tgt.shape[0]]).repeat(tgt.shape[1]).to('cuda')

        enc_state, memory_bank, lengths = self.encoder(tgt.unsqueeze(2), lengths)

        if not bptt:
            self.decoder.init_state(tgt, memory_bank, enc_state)

        dec_in = gen_seq.to(torch.int64)
        dec_out, attns = self.decoder(dec_in, memory_bank,
                                             memory_lengths=lengths,
                                             with_align=with_align)

        Q_all = self.output_layer(dec_out)

        Q_mod = Q_all.gather(2, gen_seq.to(torch.int64))

        return Q_mod * output_mask.to(torch.int64), Q_all * output_mask.to(torch.int64)

    def _compute_output_mask(self, gen_seq, eos_token):

        eos_idx = gen_seq.eq(eos_token).to(torch.int64)
        value_range = torch.arange(gen_seq.shape[0], 0, -1).to('cuda')
        output_multiplied = (eos_idx.transpose(0, 2) * value_range).transpose(0, 2)
        first_eos_idx = torch.argmax(output_multiplied, 0, keepdim=True).view(-1)

        output_mask = torch.ones(gen_seq.shape[0], gen_seq.shape[1], gen_seq.shape[2]).to('cuda')

        for row in range(0, gen_seq.shape[1]):
            output_mask[first_eos_idx[row]:, row] = 0

        return output_mask.to(torch.bool)

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

    def count_parameters(self, log=print):
        """Count number of parameters in model (& print with `log` callback).

        Returns:
            (int, int):
            * encoder side parameter count
            * decoder side parameter count
        """

        enc, dec, out_layer = 0, 0, 0
        for name, param in self.named_parameters():
            if 'encoder' in name:
                enc += param.nelement()
            elif 'decoder' in name:
                dec += param.nelement()
            else:
                out_layer += param.nelement()
        if callable(log):
            log('critic encoder: {}'.format(enc))
            log('critic decoder: {}'.format(dec))
            log('critic output layer: {}'.format(out_layer))

        return enc, dec, out_layer

class CriticV(nn.Module):

    def __init__(self, encoder, decoder, output_layer):
        super(CriticV, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_layer = output_layer

    def forward(self, tgt, gen_seq, eos_token, lengths=None, bptt=False, with_align=False):

        output_mask = self._compute_output_mask(gen_seq, eos_token)

        lengths = torch.tensor([tgt.shape[0]]).repeat(tgt.shape[1]).to('cuda')

        enc_state, memory_bank, lengths = self.encoder(tgt.unsqueeze(2), lengths)

        if not bptt:
            self.decoder.init_state(tgt, memory_bank, enc_state)

        dec_in = gen_seq.to(torch.int64)
        dec_out, attns = self.decoder(dec_in, memory_bank,
                                             memory_lengths=lengths,
                                             with_align=with_align)

        V = self.output_layer(dec_out)

        return V * output_mask.to(torch.int64)

    def _compute_output_mask(self, gen_seq, eos_token):

        eos_idx = gen_seq.eq(eos_token).to(torch.int64)
        value_range = torch.arange(gen_seq.shape[0], 0, -1).to('cuda')
        output_multiplied = (eos_idx.transpose(0, 2) * value_range).transpose(0, 2)
        first_eos_idx = torch.argmax(output_multiplied, 0, keepdim=True).view(-1)

        output_mask = torch.ones(gen_seq.shape[0], gen_seq.shape[1], gen_seq.shape[2]).to('cuda')

        for row in range(0, gen_seq.shape[1]):
            output_mask[first_eos_idx[row]:, row] = 0

        return output_mask.to(torch.bool)

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

    def count_parameters(self, log=print):
        """Count number of parameters in model (& print with `log` callback).

        Returns:
            (int, int):
            * encoder side parameter count
            * decoder side parameter count
        """

        enc, dec, out_layer = 0, 0, 0
        for name, param in self.named_parameters():
            if 'encoder' in name:
                enc += param.nelement()
            elif 'decoder' in name:
                dec += param.nelement()
            else:
                out_layer += param.nelement()
        if callable(log):
            log('critic encoder: {}'.format(enc))
            log('critic decoder: {}'.format(dec))
            log('critic output layer: {}'.format(out_layer))

        return enc, dec, out_layer

class ACNMTModel(BaseModel):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.
    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, actor, critic, train_mode, tgt_field):
        super(ACNMTModel, self).__init__(actor.encoder, actor.decoder)
        self.actor = actor
        self.critic = critic
        self.tgt_field = tgt_field
        self.train_mode = train_mode

    @property
    def encoder(self):
        return self.actor.encoder

    @property
    def decoder(self):
        return self.actor.decoder

    @property
    def generator(self):
        return self.actor.generator

    def forward(self, src, tgt, lengths, bptt=False, with_align=False,
                policy_strategy=PolicyStrategy.Categorical,
                policy_topk_sampling=-1,
                policy_sampling_temperature=1,
                policy_topp_sampling=-1):

        return self.actor(src, tgt, lengths, bptt, with_align, self.train_mode,
                          self.tgt_field, policy_strategy, policy_topk_sampling,
                          policy_sampling_temperature, policy_topp_sampling)

    def update_dropout(self, dropout):
        self.actor.update_dropout(dropout)
        self.critic.update_dropout(dropout)

    def count_parameters(self, log=print):
        """Count number of parameters in model (& print with `log` callback).

        Returns:
            (int, int):
            * encoder side parameter count
            * decoder side parameter count
        """

        actor_enc, actor_dec = self.actor.count_parameters(log)
        critic_enc, critic_dec, critic_out_layer = self.critic.count_parameters(log)

        total_actor = actor_enc + actor_dec
        total_critic = critic_enc + critic_dec + critic_out_layer

        log('* Actor: number of parameters: {}'.format(total_actor))
        log('* Critic: number of parameters: {}'.format(total_critic))
        log('* Total: number of parameters: {}'.format(total_actor + total_critic))

        return total_actor, total_critic

class A2CNMTModel(BaseModel):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.
    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, actor, critic, train_mode, tgt_field):
        super(A2CNMTModel, self).__init__(actor.encoder, actor.decoder)
        self.actor = actor
        self.critic = critic
        self.tgt_field = tgt_field
        self.train_mode = train_mode

    @property
    def encoder(self):
        return self.actor.encoder

    @property
    def decoder(self):
        return self.actor.decoder

    @property
    def generator(self):
        return self.actor.generator

    def forward(self, src, tgt, lengths, bptt=False, with_align=False,
                policy_strategy=PolicyStrategy.Categorical,
                policy_topk_sampling=-1,
                policy_sampling_temperature=1,
                policy_topp_sampling=-1):

        return self.actor(src, tgt, lengths, bptt, with_align, self.train_mode,
                          self.tgt_field, policy_strategy, policy_topk_sampling,
                          policy_sampling_temperature, policy_topp_sampling)

    def update_dropout(self, dropout):
        self.actor.update_dropout(dropout)
        self.critic.update_dropout(dropout)

    def count_parameters(self, log=print):
        """Count number of parameters in model (& print with `log` callback).

        Returns:
            (int, int):
            * encoder side parameter count
            * decoder side parameter count
        """

        actor_enc, actor_dec = self.actor.count_parameters(log)
        critic_enc, critic_dec, critic_out_layer = self.critic.count_parameters(log)

        total_actor = actor_enc + actor_dec
        total_critic = critic_enc + critic_dec + critic_out_layer

        log('* Actor: number of parameters: {}'.format(total_actor))
        log('* Critic: number of parameters: {}'.format(total_critic))
        log('* Total: number of parameters: {}'.format(total_actor + total_critic))

        return total_actor, total_critic

class LanguageModel(BaseModel):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic decoder only model.
    Currently TransformerLMDecoder is the only LM decoder implemented
    Args:
      decoder (onmt.decoders.TransformerLMDecoder): a transformer decoder
    """

    def __init__(self, encoder=None, decoder=None):
        super(LanguageModel, self).__init__(encoder, decoder)
        if encoder is not None:
            raise ValueError("LanguageModel should not be used"
                             "with an encoder")
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False, with_align=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.
        Args:
            src (Tensor): A source sequence passed to decoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on decoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.
        Returns:
            (FloatTensor, dict[str, FloatTensor]):
            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """

        if not bptt:
            self.decoder.init_state()
        dec_out, attns = self.decoder(
            src, memory_bank=None, memory_lengths=lengths,
            with_align=with_align
        )
        return dec_out, attns

    def update_dropout(self, dropout):
        self.decoder.update_dropout(dropout)

    def count_parameters(self, log=print):
        """Count number of parameters in model (& print with `log` callback).
        Returns:
            (int, int):
            * encoder side parameter count
            * decoder side parameter count
        """

        enc, dec = 0, 0
        for name, param in self.named_parameters():
            if "decoder" in name:
                dec += param.nelement()

        if callable(log):
            # No encoder in LM, seq2seq count formatting kept
            log("encoder: {}".format(enc))
            log("decoder: {}".format(dec))
            log("* number of parameters: {}".format(enc + dec))
        return enc, dec
