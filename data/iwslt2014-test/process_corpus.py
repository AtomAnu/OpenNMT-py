"""
This script processes the iwslt test parallel corpora. It only converts all html-encoded words
into their original forms.

"""

from html import unescape

with open('iwslt14-test.src.txt', 'r') as src, open('iwslt14-test-src-processed.txt', 'w') as src_processed, \
     open('iwslt14-test.tgt.txt', 'r') as tgt, open('iwslt14-test-tgt-processed.txt', 'w') as tgt_processed:

    src_lines = src.readlines()
    tgt_lines = tgt.readlines()

    for src_line, tgt_line in zip(src_lines, tgt_lines):

        src_processed.write(unescape(src_line))
        tgt_processed.write(unescape(tgt_line))