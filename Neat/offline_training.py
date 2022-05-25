import os
import sys

os.system("nohup sh -c '" + sys.executable + " evolve.py" + ">res1.txt" + "' &")
os.system("echo $! > save_pid.txt")
