""" Run video for a specific {algo, trace} pair """

import os
import sys
import signal
import subprocess
import pickle
from time import sleep
import numpy
sys.path.insert(0, "../util")
from util import master_log

########################################

RUN_SCRIPT = 'run_video.py'
RANDOM_SEED = 42
# RUN_TIME = 320  # sec
RUN_TIME = 400  # sec
MM_DELAY = 40   # millisec

########################################

def mlog(fnc="main()", msg=""):
    """ Sends message to master logger """
    master_log(pkg="run_traces.py", fnc=fnc, msg=msg)

########################################

def get_trace_files(trace_path, abr_algo):
    """ Returns a list of trace files not yet addressed """

    files_all = os.listdir(trace_path)
    files_avl = pickle.load(open("./generated_logs.pickle", "rb"))
    files_avl = files_avl[abr_algo]
    files_req = []

    for tfile in files_all:
        if tfile not in files_avl:
            files_req.append(tfile)

    mlog(fnc="get_trace_files()",
         msg="Traces unaddressed for algo {}: {}".format(abr_algo, len(files_req)))

    return files_req

########################################

def main():
    """ Main function """

    trace_path = sys.argv[1]
    abr_algo = sys.argv[2]
    process_id = sys.argv[3]
    ipaddr = sys.argv[4]

    sleep_vec = range(1, 12)  # random sleep second

    files = get_trace_files(trace_path, abr_algo)
    count_max = len(files)
    server_cmd = "exec /usr/bin/python ../rl_server/dash_server_{0}.py {0} {1}"
    mm_cmd = "mm-delay {0} mm-link 12mbps {1} /usr/bin/python {2} {3} {4} {5} {6} {7} {8}"

    for count, trace_file in enumerate(files):

        numpy.random.shuffle(sleep_vec)
        sleep_time = sleep_vec[int(process_id)]

        # ABR server command execution
        server_command = server_cmd.format(abr_algo, trace_file)
        server_process = subprocess.Popen(server_command, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, shell=True)
        mlog(msg="{}: Server command exec: {}".format(abr_algo, server_command))
        sleep(2)

        # Mahimahi command execution
        mm_command = mm_cmd.format(MM_DELAY, trace_path + trace_file, RUN_SCRIPT, ipaddr,
                                   abr_algo, RUN_TIME, process_id, trace_file, sleep_time)
        mm_process = subprocess.Popen(mm_command,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        mlog(msg="mm-delay command exec: {} of {}: {}".format(count+1, count_max, mm_command))
        (out, err) = mm_process.communicate()

        # Kill ABR algorithm server after execution (whether successful or unsuccessful)
        try:
            server_process.send_signal(signal.SIGINT)
            mlog(msg="{}: Kill signal sent to rate server successfully.".format(abr_algo))
        except Exception as exception:
            mlog(msg="{}: Sending kill signal FAILED: {}".format(abr_algo, exception))

        # Handle successful/unsuccessful cases
        if 'DONE!' in out:
            mlog(msg="{}: Process execution successful.".format(abr_algo))
        else:
            mlog(msg="{}: Process execution FAILED! Process out: {} | \
					Process error: {}".format(abr_algo, out, err))
            with open('./chrome_retry_log', 'ab') as retry_log:
                retry_log.write(abr_algo + '_' + trace_file + '\n')
                retry_log.write(out + '\n')
                retry_log.flush()

########################################

if __name__ == "__main__":
    main()

########################################
