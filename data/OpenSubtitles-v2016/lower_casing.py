with open('de-en-v2016-train-src.txt', 'r') as train_src, open('de-en-v2016-train-src-lower.txt', 'w') as train_src_lower, \
     open('de-en-v2016-train-tgt.txt', 'r') as train_tgt, open('de-en-v2016-train-tgt-lower.txt', 'w') as train_tgt_lower, \
     open('de-en-v2016-valid-src.txt', 'r') as valid_src, open('de-en-v2016-valid-src-lower.txt', 'w') as valid_src_lower, \
     open('de-en-v2016-valid-tgt.txt', 'r') as valid_tgt, open('de-en-v2016-valid-tgt-lower.txt', 'w') as valid_tgt_lower, \
     open('de-en-v2016-test-src.txt', 'r') as test_src, open('de-en-v2016-test-src-lower.txt', 'w') as test_src_lower, \
     open('de-en-v2016-test-tgt.txt', 'r') as test_tgt, open('de-en-v2016-test-tgt-lower.txt', 'w') as test_tgt_lower:

    train_src_lines = train_src.readlines()
    train_tgt_lines = train_tgt.readlines()
    valid_src_lines = valid_src.readlines()
    valid_tgt_lines = valid_tgt.readlines()
    test_src_lines = test_src.readlines()
    test_tgt_lines = test_tgt.readlines()

    for line in train_src_lines:
        train_src_lower.write(line.lower())

    for line in train_tgt_lines:
        train_tgt_lower.write(line.lower())

    for line in valid_src_lines:
        valid_src_lower.write(line.lower())

    for line in valid_tgt_lines:
        valid_tgt_lower.write(line.lower())

    for line in test_src_lines:
        test_src_lower.write(line.lower())

    for line in test_tgt_lines:
        test_tgt_lower.write(line.lower())