#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=aa8920@ic.ac.uk # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/aa8920/anaconda3/bin/:$PATH
source activate
source /vol/cuda/10.0.130/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime
conda activate a3c
cd /vol/bitbucket/aa8920/OpenNMT-py
python test_multi_gpus.py
conda deactivate