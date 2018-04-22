#!/bin/bash

#SBATCH -J nawem5270
#SBATCH -p high
#SBATCH -N 1
#SBATCH --workdir=/homedtic/rgong/neural-acoustic-word-embeddings/code_modified_joint
#SBATCH --gres=gpu:1
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=2
#SBATCH --threads-per-core=2

# Output/Error Text
# ----------------
#SBATCH -o /homedtic/rgong/neural-acoustic-word-embeddings/out/nawem5270.%N.%J.%u.out # STDOUT
#SBATCH -e /homedtic/rgong/neural-acoustic-word-embeddings/out/nawem5270.%N.%J.%u.err # STDERR


# anaconda environment
export PATH=/homedtic/rgong/anaconda2/bin:$PATH
source activate /homedtic/rgong/tensorflow-py36

for i in `seq 1 5`;
do
    python /homedtic/rgong/neural-acoustic-word-embeddings/code_modified_joint/main.py 0.0 bothX model_gpu_mtl_both_5_0_$i
done
