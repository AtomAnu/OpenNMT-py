"""Define constant values used across the project."""


class DefaultTokens(object):
    PAD = '<blank>'
    BOS = '<s>'
    EOS = '</s>'
    UNK = '<unk>'
    MASK = '<mask>'
    VOCAB_PAD = 'averyunlikelytoken'
    SENT_FULL_STOPS = [".", "?", "!"]
    PHRASE_TABLE_SEPARATOR = '|||'
    ALIGNMENT_SEPARATOR = ' ||| '


class CorpusName(object):
    VALID = 'valid'
    TRAIN = 'train'
    SAMPLE = 'sample'


class SubwordMarker(object):
    SPACER = '▁'
    JOINER = '￭'


class ModelTask(object):
    LANGUAGE_MODEL = 'lm'
    SEQ2SEQ = 'seq2seq'
    AC = 'ac'
    A2C = 'a2c'
    A3C = 'a3c'

class TrainMode:
    ACTOR = 'actor'
    CRITIC = 'critic'
    AC = 'ac'

class PolicyStrategy:
    Categorical = 'categorical'
    Epsilon = 'epsilon'
    Greedy = 'greedy'
