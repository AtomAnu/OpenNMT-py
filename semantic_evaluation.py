import sys
import numpy as np
import torch
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from datasets import load_metric
from sentence_transformers import SentenceTransformer
from torch.nn import CosineSimilarity
import math

class SemanticEval():

    def __init__(self, w_fluency, w_tlss, w_slss, gpu_id):

        self.w_fluency = w_fluency
        self.w_tlss = w_tlss
        self.w_slss = w_slss
        self.device = 'cuda:' + str(gpu_id)

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

    def eval(self, src_list, hyp_list):
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

        avg_exp_fluency, avg_exp_tlss, avg_exp_slss = 0, 0, 0

        print('1/3')
        fluency_scores = self._compute_fluency(hyp_list)
        print('2/3')
        if bool(self.w_tlss):
            tlss_scores = self._compute_token_level_semantic_similarity(src_list, hyp_list)
            avg_exp_tlss = tlss_scores.exp().mean()
        print('3/3')
        slss_scores = self._compute_sentence_level_semantic_similarity(src_list, hyp_list)

        avg_exp_fluency = fluency_scores.exp().mean()
        avg_exp_slss = slss_scores.exp().mean()

        return avg_exp_fluency, avg_exp_tlss, avg_exp_slss

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

        fluency_scores = []

        for hyp in sent_list:
            GPTLM_hyp_ids = self._GPTLM_tokenize(hyp)
            if len(GPTLM_hyp_ids) == 0:
                fluency_scores.append(0.)
            else:
                GPTLM_hyp_ids_tensor = torch.tensor([GPTLM_hyp_ids]).to(self.device)
                fluency_scores.append(self._GPTLM_forward(GPTLM_hyp_ids_tensor))

        return torch.tensor(fluency_scores).to(self.device)

    def _compute_token_level_semantic_similarity(self, src_list, sent_list):

        with torch.no_grad():
            self.bert_score_metric.add_batch(predictions=sent_list, references=src_list)
            scores = self.bert_score_metric.compute(model_type=self.bert_score_model_type,
                                                    device=self.device, use_fast_tokenizer=True)

        return torch.tensor(scores['f1']).to(self.device)

    def _compute_sentence_level_semantic_similarity(self, src_list, sent_list):

        with torch.no_grad():
            src_emb = self.sentence_transformer.encode(src_list , convert_to_tensor=True, device=self.device)
            sent_emb = self.sentence_transformer.encode(sent_list , convert_to_tensor=True, device=self.device)
            cosine_similarity = self.cos_sim(sent_emb, src_emb)

        return cosine_similarity

    # def _normalise(self, score_tensor):
    #
    #     if torch.min(score_tensor) == torch.max(score_tensor):
    #         return torch.tensor([0.5]*(len(score_tensor)-1)).to(self.device)
    #     else:
    #         norm_score_tensor = (score_tensor - torch.min(score_tensor)) / (torch.max(score_tensor) - torch.min(score_tensor))
    #         return norm_score_tensor[:-1]

os_file_path = 'data/OpenSubtitles-v2016/'
os_src_file = 'de-en-v2016-test-src-lower.txt'
os_hyp_files = ['os_a3c_actor_lower.txt',
                'os_ac_100_lower.txt',
                'os_ac_async_100_lower_lr_act.txt',
                'os_a2c_100_lower_old.txt',
                'os_a3c_100_lower.txt',
                'os_ppo_50.txt']

iw_file_path = 'data/iwslt2014-test/'
iw_src_file = 'iwslt14-test-src-processed.txt'
iw_hyp_files = ['iw_a3c_actor_lower.txt',
                'iw_ac_100_lower.txt',
                'iw_ac_async_100_lower_lr_act.txt',
                'iw_a2c_100_lower_old.txt',
                'iw_a3c_100_lower.txt',
                'iw_ppo_50.txt']

gpu_id = 0

semantic_eval = SemanticEval(1,1,1,0)

print('Evaluating on the OS test set')

with open(os_file_path + os_src_file, 'r') as os_src:

    os_src_lines = os_src.readlines()

    for file in os_hyp_files:

        with open(os_file_path + file, 'r') as os_hyp:

            os_hyp_lines = os_hyp.readlines()

            print('Evaluating: {}'.format(file))

            avg_exp_fluency, avg_exp_tlss, avg_exp_slss = semantic_eval.eval(os_src_lines, os_hyp_lines)

            print('Fluency: {}'.format(avg_exp_fluency))
            print('TLSS: {}'.format(avg_exp_tlss))
            print('SLSS: {}'.format(avg_exp_slss))

print('Evaluating on the IW test set')

with open(iw_file_path + iw_src_file, 'r') as iw_src:
    iw_src_lines = iw_src.readlines()

    for file in iw_hyp_files:
        with open(iw_file_path + file, 'r') as iw_hyp:
            iw_hyp_lines = iw_hyp.readlines()

            print('Evaluating: {}'.format(file))

            avg_exp_fluency, avg_exp_tlss, avg_exp_slss = semantic_eval.eval(iw_src_lines, iw_hyp_lines)

            print('Fluency: {}'.format(avg_exp_fluency))
            print('TLSS: {}'.format(avg_exp_tlss))
            print('SLSS: {}'.format(avg_exp_slss))


