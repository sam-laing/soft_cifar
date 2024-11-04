#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=2080-galvani
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=25G
#SBATCH -o /mnt/qb/work/oh/owl886/resnet_logs/grid_searched/%j_%x.out

# Bind the host directories to directories inside the container
singularity exec --nv \
--bind /mnt/qb/work/oh/owl886/soft_cifar:/inside_container \
--bind /mnt/qb/work/oh/owl886/datasets/cifar-10-batches-py:/mnt/qb/work/oh/owl886/datasets/cifar-10-batches-py \
--bind /mnt/qb/work/oh/owl886/soft_cifar/models:/mnt/qb/work/oh/owl886/soft_cifar/models \
/mnt/qb/work/oh/owl886/uncertainty/bud.sif python /inside_container/train_full_size.py 
