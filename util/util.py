from datetime import datetime

def master_log_create():
    with open("../run_exp/logfile.txt", "w") as log_file:
        log_file.close()

def master_log(pkg="unknown_pkg", fnc="unknown_fnc", msg=""):
    with open("../run_exp/logfile.txt", "a", 0) as log_file:
        log_file.write("{} {} {}: {}\n".format(str(datetime.now()), pkg, fnc, msg))
        log_file.close()
    return