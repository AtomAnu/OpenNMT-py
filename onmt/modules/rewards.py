import sys
import nltk.translate.bleu_score as bleu
import numpy as np
import torch
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from datasets import load_metric
from sentence_transformers import SentenceTransformer
from torch.nn import CosineSimilarity
import math

def bleu_add_1(hyp, ref):

    bleu_score = bleu.sentence_bleu([ref], hyp, smoothing_function=bleu.SmoothingFunction().method2)

    return bleu_score * 100

class UnsuperReward():

    def __init__(self, fields, w_fluency, w_tlss, w_slss, gpu_id, normalise=True):

        self.src_vocab = fields['src'].base_field.vocab
        self.tgt_vocab = fields['tgt'].base_field.vocab
        self.src_eos = fields['src'].base_field.vocab.stoi[fields['src'].base_field.eos_token]
        self.tgt_bos = fields['tgt'].base_field.vocab.stoi[fields['tgt'].base_field.init_token]
        self.tgt_eos = fields['tgt'].base_field.vocab.stoi[fields['tgt'].base_field.eos_token]
        self.tgt_unk = fields['tgt'].base_field.vocab.stoi[fields['tgt'].base_field.unk_token]
        self.tgt_pad = fields['tgt'].base_field.vocab.stoi[fields['tgt'].base_field.pad_token]
        self.w_fluency = w_fluency
        self.w_tlss = w_tlss
        self.w_slss = w_slss
        self.device = 'cuda:' + str(gpu_id)
        self.normalise = normalise
        self.special_tok_penalty = -10

        if bool(w_fluency):
            ##### For Fluency Computation
            self.GPTLM = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
            self.GPTLM.eval().to(self.device)
            # load pre-trained model tokenizer (vocabulary)
            self.GPTLM_tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

        if bool(w_tlss):
            ##### For Token-level Semantic Similarity Computation
            self.bert_score_model_type = 'xlm-roberta-base'
            self.bert_score_metric = load_metric('bertscore', keep_in_memory=True, cache_dir=sys.path[0])

        if bool(w_slss):
            ##### For Sentence-level Semantic Similarity Computation
            self.sentence_transformer = SentenceTransformer('xlm-r-bert-base-nli-stsb-mean-tokens', device=self.device)
            self.sentence_transformer.eval()
            self.cos_sim = CosineSimilarity(dim=1).to(self.device)

    def compute_reward(self, src_ids, tgt_ids, hyp_ids, process_gpu_id):
        """
        This function will compute as follows:
        1.) Fluency, Token-level Semantic Similarity and Sentence-level Semantic Similarity at each timestep
        2.) Compute the three terms above for reference translations
        3.) Perform normalisation per instance to range (0-1)
        4.) Compute pay-off function
        5.) Compute the final rewards with reward shaping

        F: Fluency
        TLSS: Token-level Semantic Similarity
        SLSS: Sentence-level Semantic Similarity

        PO = w_1 * e^(F) + w_2 * e^(TLSS) + w_3 * e^(SLSS)

        r = PO_t - PO_t-1

        :return: reward_tensor [seq_len x bs x 1]
        """

        process_device = 'cuda:' + str(process_gpu_id)

        reward_tensor = torch.zeros(hyp_ids.shape[0], hyp_ids.shape[1]).to(process_device)
        special_tok_penalty_mask = torch.ones(hyp_ids.shape[0], hyp_ids.shape[1]).to(torch.bool).to(process_device)

        for col in range(0, hyp_ids.shape[1]):
            src = ''
            ref = ''
            hyp = ''
            hyp_list = []

            for src_row in range(0, src_ids.shape[0]):

                tok_idx = int(src_ids[src_row, col])

                if tok_idx == self.src_eos:
                    break
                else:
                    tok = self.src_vocab.itos[tok_idx]
                    src += tok + ' '

            for ref_row in range(0, tgt_ids.shape[0]):

                tok_idx = int(tgt_ids[ref_row, col])

                if tok_idx == self.tgt_eos:
                    break
                else:
                    tok = self.tgt_vocab.itos[tok_idx]
                    ref += tok + ' '

            for hyp_row in range(0, hyp_ids.shape[0]):

                tok_idx = int(hyp_ids[hyp_row, col])

                if tok_idx == self.tgt_eos:
                    break
                else:
                    if tok_idx in [self.tgt_bos, self.tgt_unk, self.tgt_pad]:
                        special_tok_penalty_mask[hyp_row, col] = ~special_tok_penalty_mask[hyp_row, col]

                    tok = self.tgt_vocab.itos[tok_idx]
                    hyp += tok + ' '
                    hyp_list.append(hyp)

                    if hyp_row == hyp_ids.shape[0] - 1:
                        hyp_row += 1

            if len(hyp_list) == 0:
                continue

            if self.normalise:
                sent_list = hyp_list + [ref]
            else:
                sent_list = hyp_list

            if bool(self.w_fluency):
                # Fluency calculation
                fluency_scores = self._compute_fluency(sent_list[len(hyp_list)-1:])
            else:
                fluency_scores = torch.zeros(1).to(process_device)

            if bool(self.w_tlss):
                # TLSS calculation
                tlss_scores = self._compute_token_level_semantic_similarity([src]*len(sent_list), sent_list)
            else:
                tlss_scores = torch.zeros(1).to(process_device)

            if bool(self.w_slss):
                # SLSS calculation
                slss_scores = self._compute_sentence_level_semantic_similarity([src]*len(sent_list), sent_list)
            else:
                slss_scores = torch.zeros(1).to(process_device)

            # TODO remove the debugger below
            if col == 0:
                print('REF: {}'.format(ref))
                print('HYP: {}'.format(hyp))

                print('Fluency device: {}'.format(fluency_scores.device.index))
                print('tlss device: {}'.format(tlss_scores.device.index))
                print('slss device: {}'.format(slss_scores.device.index))

                print('Fluency: {}'.format(fluency_scores))
                print('TLSS: {}'.format(tlss_scores))
                print('SLSS: {}'.format(slss_scores))

                print('Fluency shape: {}'.format(fluency_scores.shape))
                print('TLSS shape: {}'.format(tlss_scores.shape))
                print('SLSS shape: {}'.format(slss_scores.shape))

            pay_off = self.w_fluency * fluency_scores.exp() + \
                      self.w_tlss * tlss_scores.exp() + \
                      self.w_slss * slss_scores.exp()

            reward_tensor[:hyp_row, col] = pay_off.to(process_device)

        # reward shaping
        reward_tensor[1:] -= reward_tensor[:-1].clone()

        reward_tensor = reward_tensor * special_tok_penalty_mask.to(torch.int64) + \
                        (~special_tok_penalty_mask).to(torch.int64) * self.special_tok_penalty

        return reward_tensor.unsqueeze(2).to(process_device)

    def _GPTLM_tokenize(self, word):

        return self.GPTLM_tokenizer.convert_tokens_to_ids(self.GPTLM_tokenizer.tokenize(word))

    def _GPTLM_forward(self, ids_tensor):

        with torch.no_grad():
            loss = self.GPTLM(ids_tensor, lm_labels=ids_tensor)
            if math.isnan(loss.item()):
                return 0.0
            fluency = 1 / np.exp(loss.item())
        return fluency

    def _compute_fluency(self, sent_list):

        if self.normalise:
            # if len(sent_list) < 2: print('SENT_LIST: {}'.format(sent_list))

            hyp, ref = sent_list

            ref_words = ref.strip().split()
            GPTLM_ref_ids = []

            for word in ref_words:
                GPTLM_ref_ids += self._GPTLM_tokenize(word)

            GPTLM_ref_ids_tensor = torch.tensor([GPTLM_ref_ids]).to(self.device)

            ref_fluency = self._GPTLM_forward(GPTLM_ref_ids_tensor)

        else:
            hyp = sent_list[0]

        hyp_words = hyp.strip().split()
        GPTLM_hyp_ids = []
        hyp_fluency_list = []
        for word in hyp_words:
            GPTLM_hyp_ids += self._GPTLM_tokenize(word)
            if len(GPTLM_hyp_ids) == 0:
                hyp_fluency_list.append(0.)
            else:
                GPTLM_hyp_ids_tensor = torch.tensor([GPTLM_hyp_ids]).to(self.device)
                hyp_fluency_list.append(self._GPTLM_forward(GPTLM_hyp_ids_tensor))

        if self.normalise:
            return self._normalise(torch.tensor(hyp_fluency_list + [ref_fluency]).to(self.device))
        else:
            return torch.tensor(hyp_fluency_list).to(self.device)

    def _compute_token_level_semantic_similarity(self, src_list, sent_list):

        with torch.no_grad():
            self.bert_score_metric.add_batch(predictions=sent_list, references=src_list)
            scores = self.bert_score_metric.compute(model_type=self.bert_score_model_type, device=self.device)

        if self.normalise:
            return self._normalise(torch.tensor(scores['f1']).to(self.device))
        else:
            return torch.tensor(scores['f1']).to(self.device)

    def _compute_sentence_level_semantic_similarity(self, src_list, sent_list):

        with torch.no_grad():
            src_emb = self.sentence_transformer.encode(src_list , convert_to_tensor=True, device=self.device, show_progress_bar=False)
            sent_emb = self.sentence_transformer.encode(sent_list , convert_to_tensor=True, device=self.device, show_progress_bar=False)
            cosine_similarity = self.cos_sim(sent_emb, src_emb)

        if self.normalise:
            return self._normalise(cosine_similarity)
        else:
            return cosine_similarity

    def _normalise(self, score_tensor):

        if torch.min(score_tensor) == torch.max(score_tensor):
            return torch.tensor([0.5]*(len(score_tensor)-1)).to(self.device)
        else:
            norm_score_tensor = (score_tensor - torch.min(score_tensor)) / (torch.max(score_tensor) - torch.min(score_tensor))
            return norm_score_tensor[:-1]
