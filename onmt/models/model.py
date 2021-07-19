""" Onmt NMT Model base class definition """
import torch
import torch.nn as nn
from onmt.constants import TrainMode
import numpy as np

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

    def forward(self, src, tgt, lengths, bptt=False, with_align=False, train_mode=TrainMode.ACTOR, tgt_field=None, epsilon=0):

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
                return self._step_wise_forward(src, tgt, lengths, bptt, with_align, tgt_field, epsilon)
        else:
            with torch.enable_grad():
                return self._step_wise_forward(src, tgt, lengths, bptt, with_align, tgt_field, epsilon)

    def _step_wise_forward(self, src, tgt, lengths, bptt=False, with_align=False, tgt_field=None, epsilon=0):

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

            scores = self.generator(dec_out)
            # gen_tok = torch.argmax(scores, 2).unsqueeze(2)
            gen_tok = self._choose_tok(scores, epsilon)
            gen_seq = torch.cat([gen_seq, gen_tok], dim=0)

            if step == 0:
                policy_dist = scores.exp()
            else:
                policy_dist = torch.cat([policy_dist, scores.exp()], dim=0)

        gen_seq_mask, policy_mask = self._compute_output_mask(gen_seq, tgt_field.base_field.vocab.stoi[tgt_field.base_field.eos_token])
        gen_seq = gen_seq * gen_seq_mask.to(torch.int64) + (~gen_seq_mask).to(torch.int64)
        policy_dist = policy_dist * policy_mask.to(torch.int64) + (~policy_mask).to(torch.int64)

        return gen_seq, policy_dist

    def _choose_tok(self, scores, epsilon):

        vocab_size = scores.shape[2]
        best_tok = torch.argmax(scores, 2).squeeze(0).cpu().numpy()
        gen_tok = []

        for tok in best_tok:

            epsilon_greedy_policy = self._generate_epsilon_greedy_policy(tok, vocab_size, epsilon)

            selected_tok = np.random.choice(list(range(0, vocab_size)), p=epsilon_greedy_policy)

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

    def forward(self, src, tgt, lengths, bptt=False, with_align=False, epsilon=0):

        return self.actor(src, tgt, lengths, bptt, with_align, self.train_mode, self.tgt_field, epsilon)

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

    def forward(self, src, tgt, lengths, bptt=False, with_align=False, epsilon=0):

        return self.actor(src, tgt, lengths, bptt, with_align, self.train_mode, self.tgt_field, epsilon)

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
