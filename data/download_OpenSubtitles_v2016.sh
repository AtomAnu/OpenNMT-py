#!/usr/bin/bash

##################################################################################
# This script will download the tokenized training, validation and testing datasets of
# OpenSubtitles v2016
##################################################################################

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1I6lRUfaJgtYRHUpzPRPzX7YXlRzQCf1u' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1I6lRUfaJgtYRHUpzPRPzX7YXlRzQCf1u" -O OpenSubtitles-v2016.zip && rm -rf /tmp/cookies.txt
unzip OpenSubtitles-v2016.zip