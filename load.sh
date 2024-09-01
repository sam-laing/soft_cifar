#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=2080-galvani
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:2
#SBATCH --mem=50G
#SBATCH -o /mnt/qb/work/oh/owl886/logs/%j_%x.out

# Bind the host directories to directories inside the container
singularity exec --nv \
--bind /mnt/qb/work/oh/owl886/soft_cifar:/inside_container \
--bind /mnt/qb/work/oh/owl886/datasets/cifar-10-batches-py:/mnt/qb/work/oh/owl886/datasets/cifar-10-batches-py \
/mnt/qb/work/oh/owl886/uncertainty/bud.sif python /inside_container/cifar10.py