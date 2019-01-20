import load_trace
import numpy as np
import requests
import json
import os
import a3c
from collections import Counter
import tensorflow as tf

######################################################################

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
LOG_FILE = './test_results/log_sim_rl'
TEST_TRACES = './cooked_test_traces/'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = sys.argv[1]
ABR_NN_MODEL = './model_trained_abr/pretrain_linear_reward.ckpt'

######################################################################

NUM_HOTSPOT_CHUNKS = 5
MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
VIDEO_CHUNK_LEN = 4000.0  # millisec, every time add this amount to buffer
BITRATE_LEVELS = 6
TOTAL_VIDEO_CHUNK = 49
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1
VIDEO_SIZE_FILE = './video_size_'

HOTSPOT_CHUNKS_PERCENT = 10 # percentage of chunks which are hotspot

######################################################################

class Environment:

    prefetch_decisions = []

    def __init__(self, all_cooked_time, all_cooked_bw, random_seed=RANDOM_SEED):
        assert len(all_cooked_time) == len(all_cooked_bw)

        self.pensieve_returned = []

        np.random.seed(random_seed)

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        # Keeps track of number of chunks downloaded so far
        self.video_chunk_counter = 0

        # Keeps track of normal chunks head
        self.normal_chunk_counter = 0

        self.play_buffer_size = 0.0
        self.total_buffer_size = 0.0
        self.out_of_sync_buffer = []

        # pick a random trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = {}  # in bytes
        for bitrate in xrange(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

        # Hotspot related fields
        self.hotspot_chunks = [8, 17, 25, 33, 46]

        self.num_hotspot_chunks = len(self.hotspot_chunks)
        self.last_hotspot_chunk = self.hotspot_chunks[0]
        self.next_hotspot_chunk = self.hotspot_chunks[0]
        self.hotspot_chunk_counter = 0
        self.total_buffer_size = 0 # play buffer + out-of-order buffer

        self.next_normal_bitrate = DEFAULT_QUALITY
        self.next_hotspot_bitrate = DEFAULT_QUALITY

        self.last_normal_bitrate = DEFAULT_QUALITY
        self.last_hotspot_bitrate = DEFAULT_QUALITY

        self.current_bit_rate_decision = DEFAULT_QUALITY
        self.last_bit_rate_decision = DEFAULT_QUALITY

        # Smoothness eval
        self.smoothness_eval_start_chunk = 0
        self.all_bitrate_decisions = []
        for i in xrange(TOTAL_VIDEO_CHUNK):
            self.all_bitrate_decisions.append(-1)

        self.bitrate_selected = []

######################################################################

    def reset(self):

        self.video_chunk_counter = 0
        self.normal_chunk_counter = 0

        self.play_buffer_size = 0.0
        self.total_buffer_size = 0.0
        self.out_of_sync_buffer = []

        # pick a random trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))

        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        ## Hotspot related fields
        self.hotspot_chunks = [8, 17, 25, 33, 46]

        self.num_hotspot_chunks = len(self.hotspot_chunks)
        self.last_hotspot_chunk = self.hotspot_chunks[0]
        self.next_hotspot_chunk = self.hotspot_chunks[0]
        self.hotspot_chunk_counter = 0
        self.total_buffer_size = 0 # play buffer + out-of-order buffer

        self.next_normal_bitrate = DEFAULT_QUALITY
        self.next_hotspot_bitrate = DEFAULT_QUALITY

        self.last_normal_bitrate = DEFAULT_QUALITY
        self.last_hotspot_bitrate = DEFAULT_QUALITY

        self.current_bit_rate_decision = DEFAULT_QUALITY
        self.last_bit_rate_decision = DEFAULT_QUALITY

        self.smoothness_eval_start_chunk = 0

        for i in self.all_bitrate_decisions:
            self.bitrate_selected.append(i)
        # print "Counter: {}".format(Counter(self.bitrate_selected))
        # print "Counter: {}".format(Counter(self.pensieve_returned))

        self.all_bitrate_decisions = []
        for i in xrange(TOTAL_VIDEO_CHUNK):
            self.all_bitrate_decisions.append(-1)

######################################################################

    def get_abr_rl_bitrate(self, state_data):
        try:
            abr_request = requests.post('http://10.5.20.129:9999', data=json.dumps(state_data))
            if abr_request.status_code == 200:
                bitrate_decision = int(abr_request.json()["bitrate"])
                self.pensieve_returned.append(bitrate_decision)
            else:
                bitrate_decision = -1
                print("Status code: " + abr_request.status_code)
        except Exception as exception:
            bitrate_decision = -1
            print("Exception get_abr_rl_bitrate: {}".format(exception))

        return bitrate_decision

######################################################################

    def execute_action(self, prefetch_decision):

        assert prefetch_decision >= 0 and prefetch_decision <= 1

        if self.hotspot_chunk_counter == NUM_HOTSPOT_CHUNKS:
            prefetch_decision = 0

        Environment.prefetch_decisions.append(prefetch_decision)
        # print "Prefetch decisions: {}".format(Counter(Environment.prefetch_decisions))

        # increment video chunk counter
        self.video_chunk_counter += 1
        # print "Video chunk counter: {}".format(self.video_chunk_counter)

        was_hotspot_chunk = 0
        self.last_bit_rate_decision = self.current_bit_rate_decision

        if prefetch_decision == 1:
            was_hotspot_chunk = 1
            self.current_bit_rate_decision = self.next_hotspot_bitrate
        else:
            if self.normal_chunk_counter in self.hotspot_chunks:
                was_hotspot_chunk = 1
            self.current_bit_rate_decision = self.next_normal_bitrate

        video_chunk_size = 0
        if prefetch_decision == 0:
            video_chunk_size = self.video_size[self.next_normal_bitrate][self.normal_chunk_counter]
            # video_chunk_size = self.video_size[0][self.normal_chunk_counter]

            self.all_bitrate_decisions[self.normal_chunk_counter] = self.next_normal_bitrate
            self.last_normal_bitrate = self.next_normal_bitrate
            self.normal_chunk_counter += 1
        else:
            video_chunk_size = self.video_size[self.next_hotspot_bitrate][self.next_hotspot_chunk]
            self.all_bitrate_decisions[self.next_hotspot_chunk] = self.next_hotspot_bitrate
            self.last_hotspot_chunk = self.next_hotspot_chunk
            self.hotspot_chunk_counter += 1
            if self.hotspot_chunk_counter == NUM_HOTSPOT_CHUNKS:
                self.next_hotspot_chunk = self.hotspot_chunks[self.hotspot_chunk_counter-1]
            else:
                self.next_hotspot_chunk = self.hotspot_chunks[self.hotspot_chunk_counter]
            self.last_hotspot_bitrate = self.next_hotspot_bitrate

        # print "Normal chunk counter: {}".format(self.normal_chunk_counter)
        # print "Video chunk size: {}".format(video_chunk_size)
        # print "All bitrate decisions: {}".format(self.all_bitrate_decisions)
        # print "Last normal bitrate: {}".format(self.last_normal_bitrate)
        # print "Next normal bitrate: {}".format(self.next_normal_bitrate)
        # print "Last hotspot chunk: {}".format(self.last_hotspot_chunk)
        # print "Next hotspot chunk: {}".format(self.next_hotspot_chunk)
        # print "Hotspot chunk counter: {}".format(self.hotspot_chunk_counter)
        # print "Last hotspot bitrate: {}".format(self.last_hotspot_bitrate)
        # print "Next hotspot bitrate: {}".format(self.next_hotspot_bitrate)

        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes

        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] \
                         * B_IN_MB / BITS_IN_BYTE
            duration = self.cooked_time[self.mahimahi_ptr] \
                       - self.last_mahimahi_time

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT

        # add a multiplicative noise to the delay
        delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        # rebuffer time
        # print "Delay: {}".format(delay)
        rebuf = np.maximum(delay - self.play_buffer_size, 0.0)
        # print "Rebuffering time: {}".format(rebuf)

        # update the play buffer
        self.play_buffer_size = np.maximum(self.play_buffer_size - delay, 0.0)
        # print "Play buffer size: {}".format(self.play_buffer_size)

        # add in the new chunk
        if prefetch_decision == 0:
            self.play_buffer_size += VIDEO_CHUNK_LEN
            # print "Play buffer size: {}".format(self.play_buffer_size)
        else:
            self.out_of_sync_buffer.append(self.last_hotspot_chunk)
            # print "Out of sync buffer: {}".format(self.out_of_sync_buffer)

        # reconcile buffer with out-of-sync chunks
        while len(self.out_of_sync_buffer) > 0:
            if self.out_of_sync_buffer[0] == self.normal_chunk_counter:
                self.play_buffer_size += VIDEO_CHUNK_LEN
                self.normal_chunk_counter += 1
                self.out_of_sync_buffer = self.out_of_sync_buffer[1:]
            else:
                break
        # print "Normal chunk counter: {}".format(self.normal_chunk_counter)
        # print "Play buffer size: {}".format(self.play_buffer_size)
        # print "Out of sync buffer: {}".format(self.out_of_sync_buffer)

        # compute bitrate decision array for smoothness calculation
        if self.normal_chunk_counter <= 2:
            self.smoothness_eval_start_chunk = 0
        last_in_sync_chunk = self.normal_chunk_counter
        # print "Last downloaded normal chunk: {}".format(self.smoothness_eval_start_chunk)
        # print "Last in sync chunk: {}".format(last_in_sync_chunk)

        # compute total buffer size
        self.total_buffer_size = self.play_buffer_size \
                                + (VIDEO_CHUNK_LEN * len(self.out_of_sync_buffer))
        # print "Total buffer size: {}".format(self.total_buffer_size)

        # sleep if buffer gets too large
        sleep_time = 0
        if self.total_buffer_size > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.total_buffer_size - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME
            self.play_buffer_size -= sleep_time
            self.total_buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] \
                           - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.play_buffer_size

        video_chunk_remain = TOTAL_VIDEO_CHUNK - self.video_chunk_counter
        # print "Video chunks remaining: {}".format(video_chunk_remain)

        smoothness_eval_bitrates = self.all_bitrate_decisions[self.smoothness_eval_start_chunk:last_in_sync_chunk]

        end_of_video = False
        if self.video_chunk_counter >= TOTAL_VIDEO_CHUNK-1: # or self.hotspot_chunk_counter == NUM_HOTSPOT_CHUNKS:
            end_of_video = True
            # print "End of video: {}".format(end_of_video)
            self.reset()

        next_video_chunk_sizes = []
        for i in xrange(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_size[i][self.normal_chunk_counter])
        # print "Next video chunk sizes: {}".format(next_video_chunk_sizes)

        next_hotspot_chunk_sizes = []
        for i in xrange(BITRATE_LEVELS):
            next_hotspot_chunk_sizes.append(self.video_size[i][self.next_hotspot_chunk])
        # print "Next hotspot chunk sizes: {}".format(next_hotspot_chunk_sizes)

        dist_from_hotspot_chunks = []
        for i in xrange(self.num_hotspot_chunks):
            dist_from_hotspot_chunks.append(self.hotspot_chunks[i] - self.next_hotspot_chunk)
        # print "Distance from hotspot chunks: {}".format(dist_from_hotspot_chunks)

        # print "Smoothness eval bitrates: {}".format(smoothness_eval_bitrates)

        self.last_prefetch_decision = prefetch_decision

        # -----------------------------------
        state_info_pensieve = {}
        state_info_pensieve["is_last_action_prefetch"] = prefetch_decision
        state_info_pensieve["delay"] = delay
        state_info_pensieve["sleep_time"] = sleep_time
        state_info_pensieve["buffer_size"] = return_buffer_size / MILLISECONDS_IN_SECOND
        state_info_pensieve["rebuf"] = rebuf / MILLISECONDS_IN_SECOND
        state_info_pensieve["video_chunk_size"] = video_chunk_size
        state_info_pensieve["end_of_video"] = end_of_video
        state_info_pensieve["video_chunk_remain"] = video_chunk_remain
        state_info_pensieve["last_bit_rate"] = self.last_bit_rate_decision
        state_info_pensieve["bit_rate"] = self.current_bit_rate_decision

        # state_info_pensieve['buffer'] = self.play_buffer_size / MILLISECONDS_IN_SECOND
        # state_info_pensieve['RebufferTime'] = rebuf / MILLISECONDS_IN_SECOND
        # state_info_pensieve['delay'] = delay / MILLISECONDS_IN_SECOND
        # state_info_pensieve['lastChunkSize'] = video_chunk_size
        # state_info_pensieve['videoChunkCount'] = self.video_chunk_counter
        # state_info_pensieve["nextVideoChunkIndex"] = self.normal_chunk_counter

        state_info_pensieve["next_video_chunk_sizes"] = next_video_chunk_sizes
        state_info_pensieve["is_prefetch_hotspot"] = 0

        self.next_normal_bitrate = self.get_abr_rl_bitrate(state_info_pensieve)

        # state_info_pensieve['nextVideoChunkIndex'] = self.next_hotspot_chunk
        state_info_pensieve["next_video_chunk_sizes"] = next_hotspot_chunk_sizes
        state_info_pensieve["is_prefetch_hotspot"] = 1

        self.next_hotspot_bitrate = self.get_abr_rl_bitrate(state_info_pensieve)

        # print "Next Pensieve normal bitrate: {}".format(self.next_normal_bitrate)
        # print "Next Pensieve hotspot bitrate: {}".format(self.next_hotspot_bitrate)
        # -----------------------------------

        state_data_for_action = {}

        # normal chunk state information
        state_data_for_action['delay'] = delay
        state_data_for_action['sleep_time'] = sleep_time
        if prefetch_decision == 0:
            state_data_for_action['last_bit_rate'] = self.last_normal_bitrate
        else:
            state_data_for_action['last_bit_rate'] = self.last_hotspot_bitrate
        state_data_for_action['play_buffer_size'] = self.play_buffer_size / MILLISECONDS_IN_SECOND
        state_data_for_action['rebuf'] = rebuf / MILLISECONDS_IN_SECOND
        state_data_for_action['video_chunk_size'] = video_chunk_size
        state_data_for_action['next_video_chunk_sizes'] = next_video_chunk_sizes
        state_data_for_action['end_of_video'] = end_of_video
        state_data_for_action['video_chunk_remain'] = video_chunk_remain

        # hotspot chunk state information
        state_data_for_action['was_hotspot_chunk'] = was_hotspot_chunk
        state_data_for_action['hotspot_chunks_remain'] = self.num_hotspot_chunks - self.hotspot_chunk_counter
        state_data_for_action['chunks_till_played'] = self.next_hotspot_chunk - self.normal_chunk_counter
        state_data_for_action['total_buffer_size'] = self.total_buffer_size / MILLISECONDS_IN_SECOND
        state_data_for_action['last_hotspot_bit_rate'] = self.last_hotspot_bitrate
        state_data_for_action['next_hotspot_chunk_sizes'] = next_hotspot_chunk_sizes
        state_data_for_action['dist_from_hotspot_chunks'] = dist_from_hotspot_chunks
        state_data_for_action['smoothness_eval_bitrates'] = smoothness_eval_bitrates

        self.smoothness_eval_start_chunk = last_in_sync_chunk - 1

        # abr decision state information
        state_data_for_action['normal_bitrate_pensieve'] = self.next_normal_bitrate
        state_data_for_action['hotspot_bitrate_pensieve'] = self.next_hotspot_bitrate

        return state_data_for_action

######################################################################
