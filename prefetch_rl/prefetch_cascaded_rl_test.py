""" Cascaded RL test """

import os, sys
import numpy as np
import tensorflow as tf
import load_trace
import a3c
import fixed_env_cascaded_rl as env

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['no_proxy'] = '10.5.20.129'

######################################################################

S_ABR_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_HOT_INFO = 6  # next_hs_chunk_size, num_hs_remaining, num_chunks_remaining_till_hs_chunk_played, play_buffer_size, bitrate_last_hs, dist_vector_from_hs_chunks
S_BRT_INFO = 2  # next_bit_rate, next_hs_bit_rate
S_INFO = S_ABR_INFO + S_HOT_INFO + S_BRT_INFO

ACTIONS = [0, 1]

S_LEN = 8  # take how many frames in the past
A_DIM = 2
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001

# NUM_AGENTS = 16
NUM_AGENTS = 1

TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
ENTROPY_CHANGE_INTERVAL = 20000
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
HD_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
# CHUNK_TIL_VIDEO_END_CAP = 48.0
CHUNK_TIL_VIDEO_END_CAP = 49.0
NUM_HOTSPOT_CHUNKS = 5
# NUM_HOTSPOT_CHUNKS = 7
M_IN_K = 1000.0
BITRATE_LEVELS = 6
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
DEFAULT_PREFETCH = 0 # default prefetch decision without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results_prefetch_rl_hotspot_scenarios/hotspot_scenario_{}'.format(sys.argv[1])
TEST_LOG_FOLDER = './test_results_prefetch_rl_hotspot_scenarios/hotspot_scenario_{}'.format(sys.argv[1])
TRAIN_TRACES = './cooked_traces_prefetch_rl/'
LOG_FILE = './test_results_prefetch_rl_hotspot_scenarios/hotspot_scenario_{}/log_sim_rl'.format(sys.argv[1])
TEST_TRACES = './cooked_test_traces_prefetch_rl/'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = sys.argv[1]
NN_MODEL = './model_trained_prefetch/hotdash_pretrained.ckpt'

######################################################################

def main():

    summary_dir = SUMMARY_DIR
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    log_file_dir = TEST_LOG_FOLDER
    if not os.path.exists(log_file_dir):
        os.makedirs(log_file_dir)

    TOTAL_REWARD_BITRATE = 0.0
    TOTAL_REWARD_HD_BITRATE = 0.0
    TOTAL_REWARD_REBUF = 0.0
    TOTAL_REWARD_SMOOTHNESS = 0.0
    TOTAL_REWARD = 0.0
    TOTAL_HOTSPOT_CHUNKS = 0.0

    np.random.seed(RANDOM_SEED)

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'wb')

    with tf.Session() as sess:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)

        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if NN_MODEL is not None:  # NN_MODEL is the path to file
            saver.restore(sess, NN_MODEL)
            print "Testing model restored."

        time_stamp = 0

        prefetch_decision = DEFAULT_PREFETCH
        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[prefetch_decision] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        video_count = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            state_data_for_action = net_env.execute_action(prefetch_decision)

            # normal chunk state information
            delay = state_data_for_action['delay']
            sleep_time = state_data_for_action['sleep_time']
            last_bit_rate = state_data_for_action['last_bit_rate']
            play_buffer_size = state_data_for_action['play_buffer_size']
            rebuf = state_data_for_action['rebuf']
            video_chunk_size = state_data_for_action['video_chunk_size']
            next_video_chunk_sizes = state_data_for_action['next_video_chunk_sizes']
            end_of_video = state_data_for_action['end_of_video']
            video_chunk_remain = state_data_for_action['video_chunk_remain']
            current_seq_no = state_data_for_action['current_seq_no']
            log_prefetch_decision = state_data_for_action['log_prefetch_decision']

            # hotspot chunk state information
            was_hotspot_chunk = 1.0*state_data_for_action['was_hotspot_chunk']
            TOTAL_HOTSPOT_CHUNKS += was_hotspot_chunk
            hotspot_chunks_remain = state_data_for_action['hotspot_chunks_remain']
            chunks_till_played = state_data_for_action['chunks_till_played']
            total_buffer_size = state_data_for_action['total_buffer_size']
            last_hotspot_bit_rate = state_data_for_action['last_hotspot_bit_rate']
            next_hotspot_chunk_sizes = state_data_for_action['next_hotspot_chunk_sizes']
            dist_from_hotspot_chunks = state_data_for_action['dist_from_hotspot_chunks']
            smoothness_eval_bitrates = state_data_for_action['smoothness_eval_bitrates']

            # abr decision state information
            normal_bitrate_pensieve = state_data_for_action['normal_bitrate_pensieve']
            hotspot_bitrate_pensieve = state_data_for_action['hotspot_bitrate_pensieve']

            # print len(next_video_chunk_sizes)
            # print len(next_hotspot_chunk_sizes)

            last_overall_bitrate = last_bit_rate
            if prefetch_decision == 1:
                last_overall_bitrate = last_hotspot_bit_rate

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty - smoothness
            reward_normal_br = (1.0 - was_hotspot_chunk) * (VIDEO_BIT_RATE[last_bit_rate] / M_IN_K) * 1.0
            reward_hotspot_br = was_hotspot_chunk * HD_REWARD[last_hotspot_bit_rate] * 1.0
            reward_rebuffering = REBUF_PENALTY * rebuf * 1.0
            reward_smoothness = 0.0
            if len(smoothness_eval_bitrates) > 1:
                for i in xrange(len(smoothness_eval_bitrates)-1):
                    reward_smoothness += 1.0 * SMOOTH_PENALTY * (1.0 * np.abs(VIDEO_BIT_RATE[smoothness_eval_bitrates[i+1]] -
                                               VIDEO_BIT_RATE[smoothness_eval_bitrates[i]]) / M_IN_K)

            reward = (1.0*reward_normal_br) + (1.0*reward_hotspot_br) - (1.0*reward_rebuffering) - (1.0*reward_smoothness)

            TOTAL_REWARD_BITRATE += reward_normal_br
            TOTAL_REWARD_HD_BITRATE += reward_hotspot_br
            TOTAL_REWARD_REBUF += reward_rebuffering
            TOTAL_REWARD_SMOOTHNESS += reward_smoothness
            TOTAL_REWARD += reward

            # print "reward before: {}".format(reward)

            r_batch.append(reward)

            # print "reward after: {}".format(reward)

            # log time_stamp, bit_rate, buffer_size, reward
            if not end_of_video:
                log_file.write(str(time_stamp) + '\t' +
                            str(VIDEO_BIT_RATE[last_overall_bitrate]) + '\t' +
                            str(play_buffer_size) + '\t' +
                            str(rebuf) + '\t' +
                            str(video_chunk_size) + '\t' +
                            str(delay) + '\t' +
                            str(reward) + '\t' +
                            str(log_prefetch_decision) + '\t' +
                            str(int(was_hotspot_chunk)) + '\t' +
                            str(current_seq_no) + '\n')
                log_file.flush()

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            ## Normal state S_ABR_INFO
            state[0, -1] = VIDEO_BIT_RATE[last_overall_bitrate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = play_buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :BITRATE_LEVELS] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
            ## Hotspot state S_HOT_INFO
            state[6, -1] = np.minimum(hotspot_chunks_remain, NUM_HOTSPOT_CHUNKS) / float(NUM_HOTSPOT_CHUNKS)
            state[7, -1] = np.minimum(chunks_till_played, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
            state[8, -1] = total_buffer_size / BUFFER_NORM_FACTOR
            state[9, -1] = last_hotspot_bit_rate / float(np.max(VIDEO_BIT_RATE))
            state[10, :BITRATE_LEVELS] = np.array(next_hotspot_chunk_sizes) / M_IN_K / M_IN_K
            state[11, :NUM_HOTSPOT_CHUNKS] = (np.array(dist_from_hotspot_chunks) + CHUNK_TIL_VIDEO_END_CAP) / float(2*CHUNK_TIL_VIDEO_END_CAP)

            ## Bitrate actions state S_BRT_INFO
            state[12, -1] = normal_bitrate_pensieve / float(np.max(VIDEO_BIT_RATE))
            state[13, -1] = hotspot_bitrate_pensieve / float(np.max(VIDEO_BIT_RATE))

            # compute action probability vector
            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            prefetch_decision = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            # Note: we need to discretize the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            s_batch.append(state)

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            if end_of_video:
                log_file.write('\n')
                log_file.close()
                # break

                prefetch_decision = DEFAULT_PREFETCH

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = np.zeros(A_DIM)
                action_vec[prefetch_decision] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)
                entropy_record = []

                video_count += 1

                if video_count >= len(all_file_names):
                    break

                # print "log file: {}".format(log_file)
                # print "Hot chunks: {}".format(TOTAL_HOTSPOT_CHUNKS)

                log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
                log_file = open(log_path, 'wb')

        print "Normal bitrate reward: {}".format(TOTAL_REWARD_BITRATE)
        print "Hotspot bitrate reward: {}".format(TOTAL_REWARD_HD_BITRATE)
        print "Rebuffering reward: {}".format(TOTAL_REWARD_REBUF)
        print "Smoothness reward: {}".format(TOTAL_REWARD_SMOOTHNESS)
        print "Total reward: {}".format(TOTAL_REWARD)
        print "Total hotspot chunks: {}".format(int(TOTAL_HOTSPOT_CHUNKS))

######################################################################

if __name__ == '__main__':
    main()

######################################################################
