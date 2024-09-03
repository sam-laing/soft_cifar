#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=2080-galvani
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=23:45:00
#SBATCH --gres=gpu:2
#SBATCH --mem=50G
#SBATCH -o /mnt/qb/work/oh/owl886/resnet_logs/grid_searched/%j_%x.out

# Bind the host directories to directories inside the container
singularity exec --nv \
--bind /mnt/qb/work/oh/owl886/soft_cifar:/inside_container \
--bind /mnt/qb/work/oh/owl886/datasets/CIFAR10H:/mnt/qb/work/oh/owl886/datasets/CIFAR10H \
--bind /mnt/qb/work/oh/owl886/soft_cifar/models:/mnt/qb/work/oh/owl886/soft_cifar/models \