#!/usr/bin/bash

##################################################################################
# This script will download the tokenized training, validation and testing datasets of
# OpenSubtitles v2016
##################################################################################

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1LGzSA_kz8WhIqFXyRKcjkdtBPvIAlb_K' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1LGzSA_kz8WhIqFXyRKcjkdtBPvIAlb_K" -O OpenSubtitles-v2016.zip && rm -rf /tmp/cookies.txt
unzip OpenSubtitles-v2016.zip