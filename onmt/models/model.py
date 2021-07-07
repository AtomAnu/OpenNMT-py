""" Onmt NMT Model base class definition """
import torch
import torch.nn as nn
from onmt.constants import TrainMode
import copy

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

        # TODO to be removed
        # print('dec_in: {}'.format(dec_in.shape))
        if self.max_len < dec_in.shape[0]:
            self.max_len = dec_in.shape[0]
            print('New max length: {}'.format(self.max_len))

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

class ACNMTModel(BaseModel):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.
    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, actor_encoder, actor_decoder, critic_encoder, critic_decoder, train_mode, tgt_field):
        super(ACNMTModel, self).__init__(actor_encoder, actor_decoder)
        self.encoder = actor_encoder
        self.decoder = actor_decoder
        self.critic_encoder = critic_encoder
        self.critic_decoder = critic_decoder
        self.tgt_field = tgt_field
        self.eos_token = tgt_field.eos_token

        # # create a target critic
        # self.target_critic_encoder = copy.deepcopy(self.critic_encoder)
        # self.target_critic_decoder = copy.deepcopy(self.critic_decoder)

        self.train_mode = train_mode

        # TODO remove the print line
        print('Train Mode: {}'.format(self.train_mode))

    def forward(self, src, tgt, lengths, bptt=False, with_align=False):

        # TODO remove the print line
        print('src shape: {}'.format(src.shape))
        print('src len:{}'.format(lengths.shape))

        enc_state, memory_bank, lengths = self.encoder(src, lengths)

        if not bptt:
            self.decoder.init_state(src, memory_bank, enc_state)

        if self.train_mode == TrainMode.ACTOR:

            # TODO inspect how eos is specified

            dec_in = tgt[:-1]  # exclude last target from inputs
            dec_out, attns = self.decoder(dec_in, memory_bank,
                                          memory_lengths=lengths,
                                          with_align=with_align)

            return dec_out, attns
        else:
            gen_seq = tgt[0].unsqueeze(0)
            gen_word = gen_seq
            for step in range(0, tgt.shape[0]):

                # TODO remove the print lines
                # print(self.fields['tgt'].base_field.vocab.itos[int(gen_word[:,0])])

                dec_out, attns = self.decoder(gen_word, memory_bank,
                                              step=step,
                                              memory_lengths=lengths,
                                              with_align=with_align)

                scores = self.generator(dec_out)
                gen_word = torch.argmax(scores, 2).unsqueeze(2)
                gen_seq = torch.cat([gen_seq, gen_word], dim=0)

                if step == 0:
                    policy_dist = scores.exp()
                else:
                    policy_dist = torch.cat([policy_dist, scores.exp()], dim=0)
            return gen_seq, policy_dist

    def compute_output_mask(self, gen_seq):
        # to be used when the decoder conditions on its own output
        eos_idx = gen_seq.eq(self.eos_token).to(torch.int64)
        value_range = torch.arange(gen_seq.shape[0], 0, -1)
        output_multiplied = (eos_idx.transpose(0, 2) * value_range).transpose(0, 2)
        first_eos_idx = torch.argmax(output_multiplied, 0, keepdim=True).squeeze()

        output_mask = torch.ones(gen_seq.shape[0], gen_seq.shape[1], gen_seq.shape[2])

        for row in range(0, gen_seq.shape[1]):
            output_mask[first_eos_idx[row] + 1:, row] = 0

        return output_mask


    def critic_forward(self, tgt, gen_seq, lengths=None, bptt=False, with_align=False):

        # TODO remove the print line
        print('tgt shape: {}'.format(tgt.shape))

        lengths = torch.tensor([tgt.shape[0]]).repeat(tgt.shape[1]).to('cuda')

        enc_state, memory_bank, lengths = self.critic_encoder(tgt.unsqueeze(2), lengths)

        if not bptt:
            self.critic_decoder.init_state(tgt, memory_bank, enc_state)

        dec_in = gen_seq
        dec_out, attns = self.critic_decoder(dec_in, memory_bank,
                                             memory_lengths=lengths,
                                             with_align=with_align)

        Q_all = self.critic_output_layer(dec_out)

        Q_mod = Q_all.gather(2, gen_seq.to(torch.int64))

        # TODO remove the print lines
        print('Q_mod: {}'.format(Q_mod.shape))
        print('Q_all: {}'.format(Q_all.shape))

        return Q_mod, Q_all

    # def target_critic_forward(self, ):
    #
    #     pass

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
