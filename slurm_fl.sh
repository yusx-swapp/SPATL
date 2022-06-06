#!/bin/bash

#SBATCH --nodes=1 # request one node

#SBATCH --cpus-per-task=8  # ask for 8 cpus

#SBATCH --mem=16G # Maximum amount of memory this job will be given, try to estimate this to the best of your ability. This asks for 128 GB of ram.

#SBATCH --gres=gpu:1 #If you just need one gpu, you're done, if you need more you can change the number

#SBATCH --partition=gpu #specify the gpu partition
#SBATCH --nodelist frost-6

#SBATCH --time=1-18:00:00 # ask that the job be allowed to run for 2 days, 2 hours, 30 minutes, and 2 seconds.

# everything below this line is optional, but are nice to have quality of life things

#SBATCH --output=notran_50clients.%J.out # tell it to store the output console text to a file called job.<assigned job number>.out

#SBATCH --error=notran_50clients.%J.err # tell it to store the error messages from the program (if it doesn't write them to normal console output) to a file called job.<assigned job muber>.err

#SBATCH --job-name="multi-head fl" # a nice readable name to give your job so you know what it is when you see it in the queue, instead of just numbers

# under this we just do what we would normally do to run the program, everything above this line is used by slurm to tell it what your job needs for resources

# let's load the modules we need to do what we're going to do

cd /work/LAS/jannesar-lab/yusx/MHFL

# let's load the modules we need to do what we're going to do
source /work/LAS/jannesar-lab/yusx/anaconda3/bin/activate /work/LAS/jannesar-lab/yusx/anaconda3/envs/mhfl
#source activate Model_Compression
# the commands we're running are below

nvidia-smi

python spatl_federated_learning.py \
--model=resnet20 \
--dataset=cifar10 \
--alg=spatl \
--lr=0.01 \
--batch-size=64 \
--epochs=10 \
--n_parties=100 \
--beta=0.1 \
--device='cuda' \
--datadir='./data/' \
--logdir='./logs/'  \
--noise=0 \
--sample=0.4 \
--rho=0.9 \
--partition=noniid-labeldir \
--comm_round=500 \
--init_seed=0


#\
#--partition=noniid-labeldir

#srun --nodes 1 --tasks 1 --cpus-per-task=8 --mem=64G --partition interactive --gres=gpu:1 --partition=gpu --time 8:00:00 --pty /usr/bin/bash

#chmod +x script-name-here.sh

#sbatch slurm.sh

#scontrol show job  659285
#scontrol show job 659276
#--gres=gpu:1 563573
#--gres=gpu:v100-pcie-16G:1

'''
python multi_head_federated_learning.py \
--model=resnet20 \
--dataset=cifar10 \
--alg=scaffold \
--lr=0.001 \
--batch-size=64 \
--epochs=10 \
--n_parties=30 \
--beta=0.1 \
--device='cuda' \
--datadir='./data/' \
--logdir='./logs/'  \
--noise=0 \
--sample=0.4 \
--rho=0.9 \
--comm_round=200 \
--init_seed=0

python multi_head_federated_learning.py \
--model=resnet32 \
--dataset=cifar10 \
--alg=scaffold \
--lr=0.0005 \
--batch-size=64 \
--epochs=5 \
--n_parties=50 \
--beta=0.1 \
--device='cuda' \
--datadir='./data/' \
--logdir='./logs/'  \
--noise=0 \
--sample=0.7 \
--rho=0.9 \
--comm_round=200 \
--init_seed=0

'''