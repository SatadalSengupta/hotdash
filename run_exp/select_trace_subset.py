# This module randomly selects N trace files from the set of traces, for sample experiments.

import os
import sys
import shutil
import random

TARGET_NUM = 200

files = os.listdir("../cooked_traces_all")
selected = os.listdir("../cooked_traces")
candidates = [f for f in files if f not in selected]
num = TARGET_NUM - len(selected)
select = random.sample(candidates, num)

for f in select:
    src = os.path.join("../cooked_traces_all", f)
    dst = os.path.join("../cooked_traces", f)
    shutil.copyfile(src, dst)
