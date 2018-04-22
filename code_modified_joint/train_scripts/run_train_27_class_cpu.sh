#!/bin/bash

#SBATCH -J nawe27c
#SBATCH -p high
#SBATCH -N 1
#SBATCH --workdir=/homedtic/rgong/neural-acoustic-word-embeddings/code_modified
#--nodelist=node021
#--gres=gpu:1
#SBATCH --mem=100G
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=2
#SBATCH --threads-per-core=2

# Output/Error Text
# ----------------
#SBATCH -o /homedtic/rgong/neural-acoustic-word-embeddings/out/nawe27c.%N.%J.%u.out # STDOUT
#SBATCH -e /homedtic/rgong/neural-acoustic-word-embeddings/out/nawe27c.%N.%J.%u.err # STDERR

module load Tensorflow/1.5.0-foss-2017a-Python-3.6.4

# anaconda environment
export PATH=/homedtic/rgong/anaconda2/bin:$PATH
source activate /homedtic/rgong/tensorflow-py36

python /homedtic/rgong/neural-acoustic-word-embeddings/code_modified/main.py 0.3 model_cpu

