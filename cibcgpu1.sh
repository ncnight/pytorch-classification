#!/bin/bash
#SBATCH --job-name=zfp   # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=nithin.ch10@gmail.com     # Where to send mail  
#SBATCH --ntasks=2                    # Run on a single CPU
#SBATCH --mem=64gb                     # Job memory request
#SBATCH --time=50:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/cifar10_coeff_vgg19bn_precision_all%j.log   # Standard output and error log
#SBATCH --gres=gpu:2

## Diagnostic Information
echo $CUDA_VISIBLE_DEVICES
which nvidia-smi
nvidia-smi

## Setup
source ~/.bashrc
which conda
conda activate zfp

PRECISIONS=(2 3 4 5 6 7 8 9 10 11 12 -1)

for p in ${PRECISIONS[@]}; do
    python cifar.py -a vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/vgg19_bn_$p --precision $p --best_acc_file ./secondrun.txt
done

for p in ${PRECISIONS[@]}; do
    python cifar.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-110_$p  --precision $p --best_acc_file ./secondrun.txt
done

for p in ${PRECISIONS[@]}; do
    python cifar.py -a resnet --depth 1202 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-1202_$p --precision $p --best_acc_file ./secondrun.txt
done

for p in ${PRECISIONS[@]}; do
    python cifar.py -a preresnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/preresnet-110_$p --precision $p --best_acc_file ./secondrun.txt
done

for p in ${PRECISIONS[@]}; do
    python cifar.py -a resnext --depth 29 --cardinality 8 --widen-factor 4 --schedule 150 225 --wd 5e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/resnext-8x64d_$p --precision $p --best_acc_file ./secondrun.txt
done

for p in ${PRECISIONS[@]}; do
    python cifar.py -a resnext --depth 29 --cardinality 16 --widen-factor 4 --schedule 150 225 --wd 5e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/resnext-16x64d_$p --precision $p --best_acc_file ./secondrun.txt
done

for p in ${PRECISIONS[@]}; do
    python cifar.py -a wrn --depth 28 --depth 28 --widen-factor 10 --drop 0.3 --epochs 200 --schedule 60 120 160 --wd 5e-4 --gamma 0.2 --checkpoint checkpoints/cifar10/WRN-28-10-drop_$p --precision $p --best_acc_file ./secondrun.txt
done

for p in ${PRECISIONS[@]}; do
    python cifar.py -a densenet --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/densenet-bc-100-12_$p --precision $p --best_acc_file ./secondrun.txt
done

for p in ${PRECISIONS[@]}; do
    python cifar.py -a densenet --depth 190 --growthRate 40 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/densenet-bc-L190-k40_$p --precision $p --best_acc_file ./secondrun.txt
done
