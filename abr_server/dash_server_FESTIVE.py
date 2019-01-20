#!/usr/bin/env python
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import SocketServer
import base64
import urllib
import sys
sys.path.insert(0, "../util")
import os
import logging
import json
from util import master_log, master_log_create

from collections import deque
import numpy as np
import time
from datetime import datetime
from os.path import basename as opb
from os.path import abspath as opa

######################################################################

ABR_ALGO = "FESTIVE"
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BITRATE_REWARD_MAP = {0: 0, 300: 1, 750: 2, 1200: 3, 1850: 12, 2850: 15, 4300: 20}
M_IN_K = 1000.0
DEFAULT_QUALITY = 0  # default video quality without agent
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> this number of Mbps
SMOOTH_PENALTY = 1
TOTAL_VIDEO_CHUNKS = 48
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
# in format of time_stamp bit_rate buffer_size rebuffer_time video_chunk_size download_time reward

######################################################################

def mlog(fnc="none", msg=""):
    print msg
    master_log(pkg="dash_server_FESTIVE.py", fnc=fnc, msg=msg)

######################################################################

def make_request_handler(input_dict):

    class Request_Handler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.input_dict = input_dict
            self.log_file = input_dict['log_file']
            BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length))

            mlog(fnc="do_POST()", msg="POST req data: Last request - {}, Last quality - {}, Rebuffer Time - {}".format(
                post_data['lastRequest'], post_data['lastquality'], float(post_data['RebufferTime'] - self.input_dict['last_total_rebuf'])))
            send_data = ""

            if ( 'lastquality' in post_data ):
                rebuffer_time = float(post_data['RebufferTime'] -self.input_dict['last_total_rebuf'])
                reward = \
                   VIDEO_BIT_RATE[post_data['lastquality']] / M_IN_K \
                   - REBUF_PENALTY * (post_data['RebufferTime'] - self.input_dict['last_total_rebuf']) / M_IN_K \
                   - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[post_data['lastquality']] -
                                                  self.input_dict['last_bit_rate']) / M_IN_K
                # reward = BITRATE_REWARD[post_data['lastquality']] \
                #         - 8 * rebuffer_time / M_IN_K - np.abs(BITRATE_REWARD[post_data['lastquality']] - BITRATE_REWARD_MAP[self.input_dict['last_bit_rate']])

                video_chunk_fetch_time = post_data['lastChunkFinishTime'] - post_data['lastChunkStartTime']
                video_chunk_size = post_data['lastChunkSize']
                
                # log wall_time, bit_rate, buffer_size, rebuffer_time, video_chunk_size, download_time, reward
                self.log_file.write(str(time.time()) + '\t' +
                                    str(VIDEO_BIT_RATE[post_data['lastquality']]) + '\t' +
                                    str(post_data['buffer']) + '\t' +
                                    str(float(post_data['RebufferTime'] - self.input_dict['last_total_rebuf']) / M_IN_K) + '\t' +
                                    str(video_chunk_size) + '\t' +
                                    str(video_chunk_fetch_time) + '\t' +
                                    str(reward) + '\n')
                self.log_file.flush()

                self.input_dict['last_total_rebuf'] = post_data['RebufferTime']
                self.input_dict['last_bit_rate'] = VIDEO_BIT_RATE[post_data['lastquality']]

                if ( post_data['lastRequest'] == TOTAL_VIDEO_CHUNKS ):
                    send_data = "REFRESH"
                    self.input_dict['last_total_rebuf'] = 0
                    self.input_dict['last_bit_rate'] = DEFAULT_QUALITY
                    self.log_file.write('\n')  # so that in the log we know where video ends
                    lock_path = "./locks/video_" + opb(self.input_dict['log_file_path']) + ".lock"
                    with open(lock_path, "w") as fp:
                        fp.close()
                    mlog(fnc="do_POST()", msg="Created lock file: {}".format(opa(lock_path)))

            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Content-Length', len(send_data))
            self.send_header('Access-Control-Allow-Origin', "*")
            self.end_headers()
            self.wfile.write(send_data)
            if len(send_data) > 0:
                mlog(fnc="do_POST()", msg="Response to POST req: {}".format(send_data))

        def do_GET(self):
            mlog(fnc="do_GET()", msg="Received GET REQ")
            self.send_response(200)
            #self.send_header('Cache-Control', 'Cache-Control: no-cache, no-store, must-revalidate max-age=0')
            self.send_header('Cache-Control', 'max-age=3000')
            content = "FESTIVE Server: {}".format(input_dict['log_file_path'])
            self.send_header('Content-Length', len(content))
            self.end_headers()
            # self.wfile.write("console.log('here');")
            self.wfile.write(content)
            # self.wfile.write("Simple: {}".format(datetime.now().strftime("%H-%M-%S")))
            mlog(fnc="do_GET()", msg="Response to GET req: {}".format(content))

        def log_message(self, format, *args):
            return

    return Request_Handler

######################################################################

def run(server_class=HTTPServer, port=8999, log_file_path=LOG_FILE):

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    with open(log_file_path, 'wb') as log_file:

        last_bit_rate = DEFAULT_QUALITY
        last_total_rebuf = 0 
        # input_dict = {'log_file': log_file,
        #               'last_bit_rate': last_bit_rate,
        #               'last_total_rebuf': last_total_rebuf}
        input_dict = {'log_file': log_file,
                      'last_bit_rate': last_bit_rate,
                      'last_total_rebuf': last_total_rebuf,
                      'log_file_path': log_file_path}

        handler_class = make_request_handler(input_dict=input_dict)

        #server_address = ('localhost', port)
        server_address = ('10.5.20.129', port)
        httpd = server_class(server_address, handler_class)
        mlog(fnc="run()", msg="Listening on port {}".format(port))
        httpd.serve_forever()

######################################################################

def main():
    try:
        if len(sys.argv) == 3:
            abr_algo = sys.argv[1]
            trace_file = sys.argv[2]
            mlog(fnc="main()", msg="Starting server for trace: {}".format(trace_file))
            run(log_file_path=LOG_FILE + '_' + abr_algo + '_' + trace_file)
        else:
            run()
    except Exception as e:
        mlog(fnc="main()", msg="Server function stopped. Trace: {}, Exception: {}".format(
            trace_file, e))
        raise e

######################################################################

if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            master_log_create()
        main()
    except KeyboardInterrupt:
        mlog(msg="Keyboard interrupted")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

######################################################################