""" Run experiments for all traces and all ABR algorithms """

import os
import time
import subprocess
import sys
import pickle
sys.path.insert(0, "../util")
from util import master_log, master_log_create

########################################

HOTSPOT_AWARE = False

# ABR_ALGO_CLUSTER = ["BB", "RB", "FIXED", "FESTIVE", "BOLA",
#                     "fastMPC", "robustMPC", "RL"]
ABR_ALGO_CLUSTER = ["BB", "RB", "FESTIVE",
                    "fastMPC", "robustMPC", "RL"]
NUM_OF_CHUNKS = 49

########################################

def mlog(fnc="main()", msg=""):
    """ Sends message to master logger """
    master_log(pkg="run_algo_all.py", fnc=fnc, msg=msg)

########################################

def gen_logs_avl():
    """ Generates and preserves details of already available logs """

    # Remove incomplete logs
    logs_all = os.listdir("./results")
    logs_complete = [log for log in logs_all]
    for log in logs_all:
        with open("./results/" + log, "r") as log_fp:
            lines = log_fp.readlines()
        if lines < NUM_OF_CHUNKS:
            logs_complete.remove(log)

    # Dictionary to hold already generated logs: Initialization
    gen_logs = {}
    for algo in ABR_ALGO_CLUSTER:
        gen_logs[algo] = set()

    # Dictionary to hold already generated logs: Population
    for log_file in logs_complete:
        algo_trace = log_file[4:]
        tokens = algo_trace.split("_")
        algo = tokens[0]
        trace = "_".join(tokens[1:])
        gen_logs[algo].add(trace)

    # Dictionary to hold already generated logs: Preservation
    with open("./generated_logs.pickle", "wb") as fptr:
        pickle.dump(gen_logs, fptr)

########################################

def main():
    """ Main function """

    # Generate the master log
    master_log_create()

    # Generate the list of already available logs
    gen_logs_avl()

    trace_path = '../cooked_traces/'

    with open('./chrome_retry_log', 'wb') as retry_log:
        retry_log.write('chrome retry log\n')

    os.system('sudo sysctl -w net.ipv4.ip_forward=1')

    if not os.path.exists("./locks"):
        os.makedirs("./locks")

    # ipaddr_data = json.loads(urllib.urlopen("http://ip.jsontest.com/").read())
    # ipaddr = str(ipaddr_data['ip'])
    ipaddr = "10.5.20.129"
    cmd = "python run_traces.py {} {} {} {}"
    if HOTSPOT_AWARE:
        ABR_ALGO_CLUSTER.extend(["NPF", "PFN", "PFL"])

    # Generate commands for execution for each ABR_ALGO
    commands = {}
    for process_id, abr_algo in enumerate(ABR_ALGO_CLUSTER):
        commands[abr_algo] = cmd.format(trace_path, abr_algo, process_id, ipaddr)

    # Execute subprocesses for each ABR_ALGO
    processes = {}
    for abr_algo in ABR_ALGO_CLUSTER:
        processes[abr_algo] = subprocess.Popen(commands[abr_algo],
                                               stdout=subprocess.PIPE, shell=True)
        mlog(msg="Command {} exec: {}".format(abr_algo, commands[abr_algo]))
        time.sleep(0.1)

    # Wait for each subprocess to complete
    for abr_algo in ABR_ALGO_CLUSTER:
        processes[abr_algo].wait()

    # Completion message
    mlog(msg="Execution complete for all ABR algorithms. Exiting process.")

########################################

if __name__ == "__main__":
    main()

########################################
