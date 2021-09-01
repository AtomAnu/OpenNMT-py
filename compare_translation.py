os_file_path = 'data/OpenSubtitles-v2016/'
os_src_file = 'de-en-v2016-test-src-lower.txt'
os_tgt_file = 'de-en-v2016-test-tgt-lower.txt'
os_hyp_files = ['os_ac_actor_lower.txt',
                'os_ac_100_lower.txt',
                'os_ac_async_100_lower_lr_act.txt',
                'os_a2c_100_lower_old.txt',
                'os_a3c_4gpu_10.txt',
                'os_ppo_50.txt']

with open(os_file_path + os_src_file, 'r') as os_src, \
    open(os_file_path + os_tgt_file, 'r') as os_tgt, \
    open(os_file_path + os_hyp_files[0], 'r') as os_actor, \
    open(os_file_path + os_hyp_files[1], 'r') as os_ac, \
    open(os_file_path + os_hyp_files[2], 'r') as os_async_ac, \
    open(os_file_path + os_hyp_files[3], 'r') as os_a2c, \
    open(os_file_path + os_hyp_files[4], 'r') as os_a3c, \
    open(os_file_path + os_hyp_files[5], 'r') as os_ppo:

    os_src_lines = os_src.readlines()
    os_tgt_lines = os_tgt.readlines()
    os_actor_lines = os_actor.readlines()
    os_ac_lines = os_ac.readlines()
    os_async_ac_lines = os_async_ac.readlines()
    os_a2c_lines = os_a2c.readlines()
    os_a3c_lines = os_a3c.readlines()
    os_ppo_lines = os_ppo.readlines()

    counter = 0

    for src, tgt, a, b, c, d, e, f in zip(os_src_lines, os_tgt_lines, os_actor_lines, os_ac_lines,
                                          os_async_ac_lines, os_a2c_lines, os_a3c_lines, os_ppo_lines):

        if tgt != a and tgt != b and tgt != c and tgt != d and tgt != e and tgt != f \
            and a != b and a != c and a != d and a != e and a != f \
            and b != c and b != d and b != e and b != f \
            and c != d and c != e and c != f \
            and d != e and d != f \
            and e != f:

        # if tgt != a and tgt != b and tgt != c and tgt != d and tgt != e and tgt != f \
        #     and a != b or a != c or a != d or a != e or a != f \
        #     or b != c or b != d or b != e or b != f \
        #     or c != d or c != e or c != f \
        #     or d != e or d != f \
        #     or e != f:


        #
        # if tgt != a and tgt != b and tgt != c and tgt != d and tgt != e and tgt != f \
        #     and e != a and e != b and e != c and e != d and e != f \
        #     and c not in e and e not in c:

            counter += 1

#             print('###########################')
#             print('SRC: {}'.format(src))
#             print('TGT: {}'.format(tgt))
#             print('Actor: {}'.format(a))
#             print('ACQ: {}'.format(b))
#             print('Async-ACQ: {}'.format(c))
#             print('A2C: {}'.format(d))
#             print('A3C: {}'.format(e))
#             print('PPO: {}'.format(f))
#             print('###########################')
#
# print('Total Count: {}'.format(counter))

iw_file_path = 'data/iwslt2014-test/'
iw_src_file = 'iwslt14-test-src-processed.txt'
iw_tgt_file = 'iwslt14-test-tgt-processed.txt'
iw_hyp_files = ['iw_ac_actor_lower.txt',
                'iw_ac_100_lower.txt',
                'iw_ac_async_100_lower_lr_act.txt',
                'iw_a2c_100_lower_old.txt',
                'iw_a3c_4gpu_10.txt',
                'iw_ppo_50.txt']

with open(iw_file_path + iw_src_file, 'r') as iw_src, \
    open(iw_file_path + iw_tgt_file, 'r') as iw_tgt, \
    open(iw_file_path + iw_hyp_files[0], 'r') as iw_actor, \
    open(iw_file_path + iw_hyp_files[1], 'r') as iw_ac, \
    open(iw_file_path + iw_hyp_files[2], 'r') as iw_async_ac, \
    open(iw_file_path + iw_hyp_files[3], 'r') as iw_a2c, \
    open(iw_file_path + iw_hyp_files[4], 'r') as iw_a3c, \
    open(iw_file_path + iw_hyp_files[5], 'r') as iw_ppo:

    iw_src_lines = iw_src.readlines()
    iw_tgt_lines = iw_tgt.readlines()
    iw_actor_lines = iw_actor.readlines()
    iw_ac_lines = iw_ac.readlines()
    iw_async_ac_lines = iw_async_ac.readlines()
    iw_a2c_lines = iw_a2c.readlines()
    iw_a3c_lines = iw_a3c.readlines()
    iw_ppo_lines = iw_ppo.readlines()

    counter = 0

    for src, tgt, a, b, c, d, e, f in zip(iw_src_lines, iw_tgt_lines, iw_actor_lines, iw_ac_lines,
                                          iw_async_ac_lines, iw_a2c_lines, iw_a3c_lines, iw_ppo_lines):

        if tgt != a and tgt != b and tgt != c and tgt != d and tgt != e and tgt != f \
            and a != b and a != c and a != d and a != e and a != f \
            and b != c and b != d and b != e and b != f \
            and c != d and c != e and c != f \
            and d != e and d != f \
            and e != f:

        # if tgt != a and tgt != b and tgt != c and tgt != d and tgt != e and tgt != f \
        #     and c != a and c != b and c != d and c != f \
        #     and e != a and e != b and e != c and e != d and e != f \
        #     and b != d and b != f and d != f\
        #     and a != b and a != d and a != f\
        #     and c not in e and e not in c:


            counter += 1

            print('###########################')
            print('SRC: {}'.format(src))
            print('TGT: {}'.format(tgt))
            print('Actor: {}'.format(a))
            print('ACQ: {}'.format(b))
            print('Async-ACQ: {}'.format(c))
            print('A2C: {}'.format(d))
            print('A3C: {}'.format(e))
            print('PPO: {}'.format(f))
            print('###########################')

print('Total Count: {}'.format(counter))

