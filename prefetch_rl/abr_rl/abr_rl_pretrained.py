#!/usr/bin/env python

import sys
import os
import json

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime
import numpy as np
import tensorflow as tf
import a3c

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['no_proxy'] = '10.5.20.129'

######################################################################

ABR_ALGO = "RL"
S_INFO = 6  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BITRATE_REWARD_MAP = {0: 0, 300: 1, 750: 2, 1200: 3, 1850: 12, 2850: 15, 4300: 20}
M_IN_K = 1000.0
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
TOTAL_VIDEO_CHUNKS = 48
DEFAULT_QUALITY = 0  # default video quality without agent
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> this number of Mbps
SMOOTH_PENALTY = 1
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './logs_pretrained_abr'
LOG_FILE = './logs_pretrained_abr/log'
# in format of time_stamp bit_rate buffer_size rebuffer_time video_chunk_size download_time reward
# NN_MODEL = None
NN_MODEL = './model_pretrained_abr/pretrain_linear_reward.ckpt'

######################################################################

# video chunk sizes
SIZE_VIDEO1 = [2354772, 2123065, 2177073, 2160877, 2233056, 1941625, 2157535,
               2290172, 2055469, 2169201, 2173522, 2102452, 2209463, 2275376,
               2005399, 2152483, 2289689, 2059512, 2220726, 2156729, 2039773,
               2176469, 2221506, 2044075, 2186790, 2105231, 2395588, 1972048,
               2134614, 2164140, 2113193, 2147852, 2191074, 2286761, 2307787,
               2143948, 1919781, 2147467, 2133870, 2146120, 2108491, 2184571,
               2121928, 2219102, 2124950, 2246506, 1961140, 2155012, 1433658]
SIZE_VIDEO2 = [1728879, 1431809, 1300868, 1520281, 1472558, 1224260, 1388403,
               1638769, 1348011, 1429765, 1354548, 1519951, 1422919, 1578343,
               1231445, 1471065, 1491626, 1358801, 1537156, 1336050, 1415116,
               1468126, 1505760, 1323990, 1383735, 1480464, 1547572, 1141971,
               1498470, 1561263, 1341201, 1497683, 1358081, 1587293, 1492672,
               1439896, 1139291, 1499009, 1427478, 1402287, 1339500, 1527299,
               1343002, 1587250, 1464921, 1483527, 1231456, 1364537, 889412]
SIZE_VIDEO3 = [1034108, 957685, 877771, 933276, 996749, 801058, 905515,
               1060487, 852833, 913888, 939819, 917428, 946851, 1036454,
               821631, 923170, 966699, 885714, 987708, 923755, 891604,
               955231, 968026, 874175, 897976, 905935, 1076599, 758197,
               972798, 975811, 873429, 954453, 885062, 1035329, 1026056,
               943942, 728962, 938587, 908665, 930577, 858450, 1025005,
               886255, 973972, 958994, 982064, 830730, 846370, 598850]
SIZE_VIDEO4 = [668286, 611087, 571051, 617681, 652874, 520315, 561791,
               709534, 584846, 560821, 607410, 594078, 624282, 687371,
               526950, 587876, 617242, 581493, 639204, 586839, 601738,
               616206, 656471, 536667, 587236, 590335, 696376, 487160,
               622896, 641447, 570392, 620283, 584349, 670129, 690253,
               598727, 487812, 575591, 605884, 587506, 566904, 641452,
               599477, 634861, 630203, 638661, 538612, 550906, 391450]
SIZE_VIDEO5 = [450283, 398865, 350812, 382355, 411561, 318564, 352642,
               437162, 374758, 362795, 353220, 405134, 386351, 434409,
               337059, 366214, 360831, 372963, 405596, 350713, 386472,
               399894, 401853, 343800, 359903, 379700, 425781, 277716,
               400396, 400508, 358218, 400322, 369834, 412837, 401088,
               365161, 321064, 361565, 378327, 390680, 345516, 384505,
               372093, 438281, 398987, 393804, 331053, 314107, 255954]
SIZE_VIDEO6 = [181801, 155580, 139857, 155432, 163442, 126289, 153295,
               173849, 150710, 139105, 141840, 156148, 160746, 179801,
               140051, 138313, 143509, 150616, 165384, 140881, 157671,
               157812, 163927, 137654, 146754, 153938, 181901, 111155,
               153605, 149029, 157421, 157488, 143881, 163444, 179328,
               159914, 131610, 124011, 144254, 149991, 147968, 161857,
               145210, 172312, 167025, 160064, 137507, 118421, 112270]

######################################################################

def mlog(fnc="unknown_fnc", msg=""):
    print msg
    with open("logs_pretrained_abr/abr_rl_pretrained.log", "a", 0) as log_file:
        log_file.write("{} {}: {}\n".format(str(datetime.now()), fnc, msg))
        log_file.close()
    return

######################################################################

def get_chunk_size(quality, index):
    if ( index < 0 or index > 48 ):
        return 0
    # note that the quality and video labels are inverted (i.e., quality 8 is highest and this pertains to video1)
    sizes = {5: SIZE_VIDEO1[index], 4: SIZE_VIDEO2[index], 3: SIZE_VIDEO3[index], 2: SIZE_VIDEO4[index], 1: SIZE_VIDEO5[index], 0: SIZE_VIDEO6[index]}
    return sizes[quality]

######################################################################

def make_request_handler(input_dict):

    class Request_Handler(BaseHTTPRequestHandler):

        train_counter = 0
        last_hotspot_state = np.array(0)
        last_normal_state = np.array(0)
        prefetch_decisions = []
        time_stamp = 0

        def __init__(self, *args, **kwargs):
            self.input_dict = input_dict
            self.sess = input_dict['sess']
            self.log_file = input_dict['log_file']
            self.actor = input_dict['actor']
            self.critic = input_dict['critic']
            self.saver = input_dict['saver']
            self.s_batch = input_dict['s_batch']
            self.a_batch = input_dict['a_batch']
            self.r_batch = input_dict['r_batch']
            self.entropy_record = input_dict['entropy_record']

            BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

        def do_POST(self):

            content_length = int(self.headers['Content-Length'])
            env_post_data = json.loads(self.rfile.read(content_length))

            # mlog(fnc="do_POST()", msg="POST req data: Last request - {}, Last quality - {}, Rebuffer Time - {}".format(
            #     post_data['lastRequest'], post_data['lastquality'], float(post_data['RebufferTime'] - self.input_dict['last_total_rebuf'])))
            send_data = ""

            if ( 'pastThroughput' in env_post_data ):
                # @Hongzi: this is just the summary of throughput/quality at the end of the load
                # so we don't want to use this information to send back a new quality
                mlog(fnc="do_POST()", msg="Past throughput is present in post_data, \
                        not using this information to send back quality")
            else:
                
                # Get params according to rl_test.py in original Pensieve code
                delay = env_post_data["delay"]
                sleep_time = env_post_data["sleep_time"]
                buffer_size = env_post_data["buffer_size"]
                rebuf = env_post_data["rebuf"]
                video_chunk_size = env_post_data["video_chunk_size"]
                next_video_chunk_sizes = env_post_data["next_video_chunk_sizes"]
                end_of_video = env_post_data["end_of_video"]
                video_chunk_remain = env_post_data["video_chunk_remain"]

                # Get additional params to differentiate between hotspot y/n cases
                bit_rate = env_post_data["bit_rate"]
                last_bit_rate = env_post_data["last_bit_rate"]
                is_last_action_prefetch = env_post_data["is_last_action_prefetch"]
                is_prefetch_hotspot = env_post_data["is_prefetch_hotspot"]

                Request_Handler.time_stamp += delay  # in ms
                Request_Handler.time_stamp += sleep_time  # in ms

                # rebuffer_time = float(post_data['RebufferTime'] - self.input_dict['last_total_rebuf'])

                # # --linear reward--
                # reward = VIDEO_BIT_RATE[post_data['lastquality']] / M_IN_K \
                #         - REBUF_PENALTY * rebuffer_time / M_IN_K \
                #         - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[post_data['lastquality']] -
                #                                   self.input_dict['last_bit_rate']) / M_IN_K

                # --log reward--
                # log_bit_rate = np.log(VIDEO_BIT_RATE[post_data['lastquality']] / float(VIDEO_BIT_RATE[0]))   
                # log_last_bit_rate = np.log(self.input_dict['last_bit_rate'] / float(VIDEO_BIT_RATE[0]))
                # reward = log_bit_rate \
                #          - 4.3 * rebuffer_time / M_IN_K \
                #          - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

                # --hd reward--
                # reward = BITRATE_REWARD[post_data['lastquality']] \
                #         - 8 * rebuffer_time / M_IN_K - np.abs(BITRATE_REWARD[post_data['lastquality']] - BITRATE_REWARD_MAP[self.input_dict['last_bit_rate']])

                # Linear reward
                reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                               VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

                # self.input_dict['last_bit_rate'] = VIDEO_BIT_RATE[post_data['lastquality']]
                # self.input_dict['last_total_rebuf'] = post_data['RebufferTime']

                self.r_batch.append(reward)

                # custom: append last state
                if Request_Handler.train_counter > 0:
                    if is_last_action_prefetch == 1:
                        self.s_batch.append(Request_Handler.last_hotspot_state)
                    else:
                        self.s_batch.append(Request_Handler.last_normal_state)

                # retrieve previous state
                if len(self.s_batch) == 0:
                    state = [np.zeros((S_INFO, S_LEN))]
                else:
                    state = np.array(self.s_batch[-1], copy=True)

                # compute bandwidth measurement
                # video_chunk_fetch_time = post_data['delay']
                # video_chunk_size = post_data['lastChunkSize']

                # compute number of video chunks left
                # video_chunk_remain = TOTAL_VIDEO_CHUNKS - post_data['videoChunkCount']

                # dequeue history record
                state = np.roll(state, -1, axis=1)
                # print "roll: {}, shape: {}".format(type(state), state.shape)

                # next_video_chunk_sizes = []
                # for i in xrange(A_DIM):
                #     next_video_chunk_sizes.append(get_chunk_size(i, post_data['nextVideoChunkIndex']))

                # this should be S_INFO number of terms
                # try:
                #     state[0, -1] = VIDEO_BIT_RATE[post_data['lastquality']] / float(np.max(VIDEO_BIT_RATE))
                #     state[1, -1] = post_data['buffer'] / BUFFER_NORM_FACTOR
                #     state[2, -1] = float(video_chunk_size) / float(video_chunk_fetch_time) / M_IN_K  # kilo byte / ms
                #     state[3, -1] = float(video_chunk_fetch_time) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
                #     state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
                #     state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
                
                #     print "Video bitrate: {}".format(state[0, -1])
                #     print "Buffer: {}".format(state[1, -1])
                #     print "Throughput: {}".format(state[2, -1])
                #     print "Download duration: {}".format(state[3, -1])
                #     print "Next video chunk sizes: {}".format(state[4, :A_DIM])
                #     print "Video chunks remaining: {}".format(state[5, -1])
                #     print "\n"
                
                # except ZeroDivisionError:
                #     # this should occur VERY rarely (1 out of 3000), should be a dash issue
                #     # in this case we ignore the observation and roll back to an eariler one
                #     if len(self.s_batch) == 0:
                #         state = [np.zeros((S_INFO, S_LEN))]
                #     else:
                #         state = np.array(self.s_batch[-1], copy=True)

                # this should be S_INFO number of terms
                try:
                    state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
                    state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
                    state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
                    state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
                    state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
                    state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

                except ZeroDivisionError:
                    # this should occur VERY rarely (1 out of 3000), should be a dash issue
                    # in this case we ignore the observation and roll back to an eariler one
                    if len(self.s_batch) == 0:
                        state = [np.zeros((S_INFO, S_LEN))]
                    else:
                        state = np.array(self.s_batch[-1], copy=True)

                # log wall_time, bit_rate, buffer_size, rebuffer_time, video_chunk_size, download_time, reward
                # self.log_file.write(str(time.time()) + '\t' +
                #                     str(VIDEO_BIT_RATE[post_data['lastquality']]) + '\t' +
                #                     str(post_data['buffer']) + '\t' +
                #                     str(rebuffer_time / M_IN_K) + '\t' +
                #                     str(video_chunk_size) + '\t' +
                #                     str(video_chunk_fetch_time) + '\t' +
                #                     str(reward) + '\n')
                # self.log_file.flush()
                # print "state construct: {}, shape: {}".format(type(state), state.shape)

                action_prob = self.actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
                action_cumsum = np.cumsum(action_prob)
                bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                # Note: we need to discretize the probability into 1/RAND_RANGE steps,
                # because there is an intrinsic discrepancy in passing single state and batch states

                self.entropy_record.append(a3c.compute_entropy(action_prob[0]))

                # send data to html side
                # send_data = str(bit_rate)
                send_data = json.dumps({"bitrate": bit_rate})
                mlog(fnc="do_POST()", msg="Bitrate decision: {}".format(bit_rate))

                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.send_header('Content-Length', len(send_data))
                self.send_header('Access-Control-Allow-Origin', "*")
                self.end_headers()
                self.wfile.write(send_data)

                # record [state, action, reward]
                # put it here after training, notice there is a shift in reward storage

                if is_prefetch_hotspot == 1:
                    Request_Handler.last_hotspot_state = state
                    Request_Handler.prefetch_decisions.append(0)
                else:
                    Request_Handler.last_normal_state = state
                    Request_Handler.prefetch_decisions.append(1)

                # self.s_batch.append(state)
                # print "batch append: {}, shape: {}".format(type(state), state.shape)

                Request_Handler.train_counter += 1
                # print "Train counter: {}".format(Request_Handler.train_counter)
                # print "Prefetch Decisions: {}".format(Counter(Request_Handler.prefetch_decisions))

        def do_GET(self):
            print "do_GET"
            mlog(fnc="do_GET()", msg="Received GET REQ")
            self.send_response(200)
            #self.send_header('Cache-Control', 'Cache-Control: no-cache, no-store, must-revalidate max-age=0')
            self.send_header('Cache-Control', 'max-age=3000')
            content = "RL Server No Training: {}".format(input_dict['log_file_path'])
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
            mlog(fnc="do_GET()", msg="Response to GET req: {}".format(content))

        def log_message(self, format, *args):
            return

    return Request_Handler

######################################################################

def run(server_class=HTTPServer, port=9999, log_file_path=LOG_FILE):

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    with tf.Session() as sess, open(log_file_path, 'wb') as log_file:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            mlog(fnc="run()", msg="Pre-trained RL model restored.")

        init_action = np.zeros(A_DIM)
        init_action[DEFAULT_QUALITY] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [init_action]
        r_batch = []
        entropy_record = []

        train_counter = 0

        last_bit_rate = DEFAULT_QUALITY
        last_total_rebuf = 0
        # need this storage, because observation only contains total rebuffering time
        # we compute the difference to get rebuffering due to last decision

        video_chunk_count = 0

        input_dict = {'sess': sess, 'log_file': log_file,
                      'actor': actor, 'critic': critic,
                      'saver': saver, 'train_counter': train_counter,
                      'last_bit_rate': last_bit_rate,
                      'last_total_rebuf': last_total_rebuf,
                      'video_chunk_count': video_chunk_count,
                      's_batch': s_batch, 'a_batch': a_batch, 'r_batch': r_batch,
                      'entropy_record': entropy_record,
                      'log_file_path': log_file_path}

        # interface to abr_rl server
        handler_class = make_request_handler(input_dict=input_dict)

        ipaddress = '10.5.20.129'
        # ipaddress = '10.5.20.61'
        server_address = (ipaddress, port)
        httpd = server_class(server_address, handler_class)
        mlog(fnc="run()", msg="Listening on IP {}, port {}".format(ipaddress, port))
        httpd.serve_forever()

######################################################################

def main():
    
    try:
        mlog(fnc="main()", msg="Starting ABR RL server")
        run(log_file_path="logs_pretrained_abr/abr_rl_results.txt")
   
    except Exception as e:
        mlog(fnc="main()", msg="Server function stopped. Exception: {}".format(e))
        raise e

######################################################################

if __name__ == "__main__":
    
    try:
        with open("logs_pretrained_abr/abr_rl_pretrained.log", "w") as fp:
            fp.close()
        main()
    except KeyboardInterrupt:
        mlog(msg="Keyboard interrupted")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

######################################################################
