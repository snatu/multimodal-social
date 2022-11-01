#!/bin/bash
#
#SBATCH -p gpu_low
#SBATCH --gres=gpu:1  # Use GPU
#SBATCH --mem 56GB   # memory pool for all cores
#SBATCH -t 1-00:00    # time (D-HH:MM)
#SBATCH -o /home/shounak_rtml/11777/MTAG/results/%j.out        # STDOUT. %j specifies JOB_ID.
#SBATCH -e /home/shounak_rtml/11777/MTAG/results/%j.err        # STDERR. See the first link for more options.
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dummyblah123@gmail.com
#SBATCH --exclude=compute-0-33,compute-1-37,compute-1-13,compute-0-37,compute-1-25,compute-1-9

# run this with: sbatch -p gpu_low ./blah.sh
# echo "hi"
# sleep 2
# source activate fairseq

cd /home/shounak_rtml/11777/MTAG

ulimit -v unlimited
singularity exec --nv -B /home/shounak_rtml/11777/awilf/.local/python3.7/site-packages:/home/awilf/.local/lib/python3.7/site-packages,/home/shounak_rtml/11777/MTAG,/home/shounak_rtml/11777/Standard-Grid,/home/shounak_rtml/11777/CMU-MultimodalSDK,/home/shounak_rtml/11777/Social-IQ blah4.sif bash run.sh


