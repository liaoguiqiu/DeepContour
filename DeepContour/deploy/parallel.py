import os
import sys
import subprocess


subprocess.run("python -m deploy.DeepAutoJson_jelly_contact & \
python -m deploy.DeepAutoJson_jelly_contact", shell=True)
subprocess.run("python -m deploy.DeepAutoJson_jelly_contact", shell=True)
# for cmd in ['-m deploy.DeepAutoJson_jelly_contact']:
#     subprocess.Popen(["python", cmd], shell=True)