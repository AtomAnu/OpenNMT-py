import nltk.translate.bleu_score as bleu

def bleu_add_1(hyp, ref):

    bleu_score = bleu.sentence_bleu([ref], hyp, smoothing_function=bleu.SmoothingFunction().method2)

    return bleu_score