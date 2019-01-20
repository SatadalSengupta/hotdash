""" Video playback for a specific algo-trace pair """

import sys
import signal
from time import sleep
from os.path import abspath as opa, exists as ope
from os import system, remove as orm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from pyvirtualdisplay import Display
sys.path.insert(0, "../util")
from util import master_log

########################################

# TO RUN: download https://pypi.python.org/packages/source/s/selenium/selenium-2.39.0.tar.gz
# run sudo apt-get install python-setuptools
# run sudo apt-get install xvfb
# after untar, run sudo python setup.py install
# follow directions here: https://pypi.python.org/pypi/PyVirtualDisplay to install pyvirtualdisplay

# For chrome, need chrome driver: https://code.google.com/p/selenium/wiki/ChromeDriver
# chromedriver variable should be path to the chromedriver
# the default location for firefox is /usr/bin/firefox and chrome binary is /usr/bin/google-chrome
# if they are at those locations, don't need to specify

########################################

# Brower dispay (True) or suppressed display (False)
BROWSER_DISPLAY = False

# Run time till when video playback continues
FIXED_RUN_TIME = 300 # 5 mins
BOLA_RUN_TIME = 360 # 6 mins
ABR_RUN_TIME = 600 # 10 mins

########################################

def mlog(abr_algo, trace_file, msg):
    """ Sends message to master logger """
    master_log(pkg="run_video.py", fnc="{} {}".format(abr_algo, trace_file), msg=msg)

########################################

def timeout_handler(dummy_signum, dummy_frame):
    """ Timeout exception handler """
    raise Exception("Timeout encountered!")

########################################

def main():
    """ Main function """

    ipaddr = sys.argv[1]
    abr_algo = sys.argv[2]
    dummy_run_time = int(sys.argv[3])
    process_id = sys.argv[4]
    trace_file = sys.argv[5]
    sleep_time = sys.argv[6]

    # Prevent multiple process from being synchronized
    sleep(int(sleep_time))

    # Generate URL
    url = "http://{}/myindex_{}.html".format(ipaddr, abr_algo)
    mlog(abr_algo=abr_algo, trace_file=trace_file,
         msg="Server URL: {}".format(url))

    # Set timeout alarm and handler
    signal.signal(signal.SIGALRM, timeout_handler)

    # Set timeout value depending on what algorithm is being used
    # FIXED and BOLA take longer time to playback (from experience)
    if abr_algo == "FIXED":
        curr_runtime = FIXED_RUN_TIME
    elif abr_algo == "BOLA":
        curr_runtime = BOLA_RUN_TIME
    else:
        curr_runtime = ABR_RUN_TIME

    # Timeout set after current run time as decided
    signal.alarm(curr_runtime)
    mlog(abr_algo=abr_algo, trace_file=trace_file,
         msg="Run time alarm set at {}.".format(curr_runtime))

    try:
        # Copy over the chrome user dir
        default_chrome_user_dir = "../abr_browser_dir/chrome_data_dir"
        chrome_user_dir = "/tmp/chrome_user_dir_id_{}".format(process_id)
        system("rm -r {}".format(chrome_user_dir))
        system("cp -r {} {}".format(default_chrome_user_dir, chrome_user_dir))

        # Display the page in browser: Yes/No
        if BROWSER_DISPLAY:
            mlog(abr_algo=abr_algo, trace_file=trace_file,
                 msg="Started display on browser.")
        else:
            display = Display(visible=0, size=(800, 600))
            display.start()
            mlog(abr_algo=abr_algo, trace_file=trace_file,
                 msg="Started supressed display.")

        # Initialize Chrome driver
        options = Options()
        chrome_driver = "../abr_browser_dir/chromedriver"
        options.add_argument("--user-data-dir={}".format(chrome_user_dir))
        options.add_argument("--ignore-certificate-errors")
        driver = webdriver.Chrome(chrome_driver, chrome_options=options)

        # Run Chrome
        driver.set_page_load_timeout(10)
        driver.get(url)
        mlog(abr_algo=abr_algo, trace_file=trace_file,
             msg="Video playback started.")

        # Sleep until lock is created by ABR server
        lock_file_path = "./locks/video_log_" + abr_algo + "_" + trace_file + ".lock"
        mlog(abr_algo=abr_algo, trace_file=trace_file,
             msg="Looking for log file: {}".format(opa(lock_file_path)))
        sleep(200) # running time of video is 193s
        while not ope(lock_file_path):
            mlog(abr_algo=abr_algo, trace_file=trace_file,
                 msg="Not found lock file, going back to sleep for 20 secs.")
            sleep(20)

        # Remove lock after it's existence is known
        orm(lock_file_path)

        # Quit the video playback
        driver.quit()
        if BROWSER_DISPLAY:
            mlog(abr_algo=abr_algo, trace_file=trace_file,
                 msg="Stopped Chrome driver.")
        else:
            display.stop()
            mlog(abr_algo=abr_algo, trace_file=trace_file,
                 msg="Stopped supressed display and Chrome driver.")

        print 'DONE!'

    except Exception as exception1:
        mlog(abr_algo=abr_algo, trace_file=trace_file,
             msg="Exception: {}".format(exception1))
        if not BROWSER_DISPLAY:
            try:
                display.stop()
                mlog(abr_algo=abr_algo, trace_file=trace_file,
                     msg="Exception Handler: Stopped suppressed display.")
            except Exception as exception2:
                mlog(abr_algo=abr_algo, trace_file=trace_file,
                     msg="Exception Again (Suppressed display): {}".format(exception2))
        try:
            driver.quit()
            mlog(abr_algo=abr_algo, trace_file=trace_file,
                 msg="Exception Handler (Chrome driver): Quit Chrome driver.")
        except Exception as exception3:
            mlog(abr_algo=abr_algo, trace_file=trace_file,
                 msg="Exception Again (Chrome driver): {}".format(exception3))

########################################

if __name__ == "__main__":
    main()

########################################
