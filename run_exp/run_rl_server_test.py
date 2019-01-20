import os
import sys
import subprocess

trace_file = "test_norway_train.vestby-oslo-report.2011-02-11_1729CET.log_120"

try:	
	command = 'exec /usr/bin/python ../rl_server/rl_server_no_training.py ' + trace_file
	proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
	outs, errs = proc.communicate(15)
	print outs
	print errs

except Exception as e:
    proc.kill()
    outs, errs = proc.communicate()
    print outs
    print errs
    print e
