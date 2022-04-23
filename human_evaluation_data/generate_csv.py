import pandas as pd
import numpy as np

src_fname = 'iwslt14-test.src.txt'
tgt_fname = 'iwslt14-test.tgt.txt'
actor_fname = 'iw_ac_actor_lower.txt'
sac_bleu_fname = 'sac.test'
acq_bleu_fname = 'iw_ac_bleu_100_lower_lr.txt'
acq_async_fname = 'iw_ac_async_100_lower_lr_act.txt'

output_csv = 'iwslt_results.csv'

with open(src_fname, 'r') as src, \
    open(tgt_fname, 'r') as tgt, \
    open(actor_fname, 'r') as actor, \
    open(sac_bleu_fname, 'r') as sac, \
    open(acq_bleu_fname, 'r') as acq_bleu, \
    open(acq_async_fname, 'r') as acq_async:

    src_lines = src.readlines()
    tgt_lines = tgt.readlines()
    actor_lines = actor.readlines()
    sac_lines = sac.readlines()
    acq_bleu_lines = acq_bleu.readlines()
    acq_async_lines = acq_async.readlines()

    data = {'SRC': src_lines[:50],
            'TGT': tgt_lines[:50],
            'Transformer': actor_lines[:50],
            'SAC-BLEU': sac_lines[:50],
            'ACQ-BLEU': acq_bleu_lines[:50],
            'ACQ-Async': acq_async_lines[:50]}

    df = pd.DataFrame(data)

    df.to_csv(output_csv)