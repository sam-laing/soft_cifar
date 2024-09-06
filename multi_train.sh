#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=2080-galvani
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00  
#SBATCH --gres=gpu:2
#SBATCH --mem=50G
#SBATCH -o /mnt/qb/work/oh/owl886/resnet_logs/grid_searched/%j_%x.out

# Bind the host directories to directories inside the container
BIND_DIRS="/mnt/qb/work/oh/owl886/soft_cifar:/inside_container,\
/mnt/qb/work/oh/owl886/datasets/CIFAR10H:/mnt/qb/work/oh/owl886/datasets/CIFAR10H,\
/mnt/qb/work/oh/owl886/soft_cifar/models:/mnt/qb/work/oh/owl886/soft_cifar/models"

CONTAINER_PATH="/mnt/qb/work/oh/owl886/uncertainty/bud.sif"
SCRIPT_PATH="/inside_container/train.py"

# Define an array of commands to run
commands=(
    "python $SCRIPT_PATH --unc_method sngp"
    "python $SCRIPT_PATH --unc_method sngp --do_augmentation n"
    "python $SCRIPT_PATH --unc_method sngp --mixup 0.2 --mixup_prob 0.15"
    "python $SCRIPT_PATH --unc_method sngp --cutmix 0.2 --cutmix_prob 0.15"
    "python $SCRIPT_PATH --unc_method sngp --mixup 0.2 --mixup_prob 0.15 --do_augmentation n"
    "python $SCRIPT_PATH --unc_method sngp --cutmix 0.2 --cutmix_prob 0.15 --do_augmentation n"
    "python $SCRIPT_PATH --unc_method sngp --hard n"
    "python $SCRIPT_PATH --unc_method sngp --do_augmentation n --hard n"
    "python $SCRIPT_PATH --unc_method sngp --mixup 0.2 --mixup_prob 0.15 --hard n"
    "python $SCRIPT_PATH --unc_method sngp --cutmix 0.2 --cutmix_prob 0.15 --hard n"
    "python $SCRIPT_PATH --unc_method sngp --mixup 0.2 --mixup_prob 0.15 --do_augmentation n --hard n"
    "python $SCRIPT_PATH --unc_method sngp --cutmix 0.2 --cutmix_prob 0.15 --do_augmentation n --hard n"
)

# Run all commands sequentially
for cmd in "${commands[@]}"; do
    singularity exec --nv --bind $BIND_DIRS $CONTAINER_PATH $cmd
done
