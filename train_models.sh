#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=2080-galvani
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=2:30:00
#SBATCH --gres=gpu:2
#SBATCH --mem=50G
#SBATCH -o /mnt/qb/work/oh/owl886/resnet_logs/grid_searched/%j_%x.out

# Bind the host directories to directories inside the container
singularity exec --nv \
--bind /mnt/qb/work/oh/owl886/soft_cifar:/inside_container \
--bind /mnt/qb/work/oh/owl886/datasets/CIFAR10H:/mnt/qb/work/oh/owl886/datasets/CIFAR10H \
--bind /mnt/qb/work/oh/owl886/soft_cifar/models:/mnt/qb/work/oh/owl886/soft_cifar/models \
/mnt/qb/work/oh/owl886/uncertainty/bud.sif python /inside_container/train.py --unc_method basic --dropout 0.05  




#/mnt/qb/work/oh/owl886/uncertainty/bud.sif python /inside_container/train.py --unc_method duq 
#/mnt/qb/work/oh/owl886/uncertainty/bud.sif python /inside_container/train.py --unc_method duq --do_augmentation n
#/mnt/qb/work/oh/owl886/uncertainty/bud.sif python /inside_container/train.py --unc_method duq --mixup 0.2 --mixup_prob 0.15
#/mnt/qb/work/oh/owl886/uncertainty/bud.sif python /inside_container/train.py --unc_method duq --cutmix 0.2 --cutmix_prob 0.15
#/mnt/qb/work/oh/owl886/uncertainty/bud.sif python /inside_container/train.py --unc_method duq --mixup 0.2 --mixup_prob 0.15 --do_augmentation n 
#/mnt/qb/work/oh/owl886/uncertainty/bud.sif python /inside_container/train.py --unc_method duq --cutmix 0.2 --cutmix_prob 0.15 --do_augmentation n
#/mnt/qb/work/oh/owl886/uncertainty/bud.sif python /inside_container/train.py --unc_method duq --hard n
#/mnt/qb/work/oh/owl886/uncertainty/bud.sif python /inside_container/train.py --unc_method duq --do_augmentation n --hard n
#/mnt/qb/work/oh/owl886/uncertainty/bud.sif python /inside_container/train.py --unc_method duq --mixup 0.2 --mixup_prob 0.15 --hard n
#/mnt/qb/work/oh/owl886/uncertainty/bud.sif python /inside_container/train.py --unc_method duq --cutmix 0.2 --cutmix_prob 0.15 --hard n
#/mnt/qb/work/oh/owl886/uncertainty/bud.sif python /inside_container/train.py --unc_method duq --mixup 0.2 --mixup_prob 0.15 --do_augmentation n --hard n
#/mnt/qb/work/oh/owl886/uncertainty/bud.sif python /inside_container/train.py --unc_method duq --cutmix 0.2 --cutmix_prob 0.15 --do_augmentation n --hard n


