#!/bin/bash
#SBATCH --job-name=zfp-profiling   # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=nithin.ch10@gmail.com     # Where to send mail  
#SBATCH --ntasks=2                    # Run on a single CPU
#SBATCH --mem=64gb                     # Job memory request
#SBATCH --time=50:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/cifar10_timing%j.log   # Standard output and error log
#SBATCH --gres=gpu:2

## Diagnostic Information
echo $CUDA_VISIBLE_DEVICES
which nvidia-smi
nvidia-smi

## Setup
# source ~/.bashrc
which conda
conda activate zfp-nn

mkdir results
cd results 
mkdir timing
mkdir profiling
cd ..

COMPRESSIONS=(npy jpg zfp)

for c in ${COMPRESSIONS[@]}; do
    nvprof -o results/profiling/vgg19-$c python profiling.py -a vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/vgg19_bn_$c --compression $c
done

for c in ${COMPRESSIONS[@]}; do
     nvprof -o results/profiling/resnet110-$c python profiling.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-110_$c --compression $c
done

for c in ${COMPRESSIONS[@]}; do
    nvprof -o results/profiling/resnet1202-$c python profiling.py -a resnet --depth 1202 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-1202_$c --compression $c
done

for c in ${COMPRESSIONS[@]}; do
    nvprof -o results/profiling/preresnet110-$c python profiling.py -a preresnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/preresnet-110_$c --compression $c
done

for c in ${COMPRESSIONS[@]}; do
    nvprof -o results/profiling/resnext29-8$c python profiling.py -a resnext --depth 29 --cardinality 8 --widen-factor 4 --schedule 150 225 --wd 5e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/resnext-8x64d_$c --compression $c
done

for c in ${COMPRESSIONS[@]}; do
    nvprof -o results/profiling/resnext29-16-$c python profiling.py -a resnext --depth 29 --cardinality 16 --widen-factor 4 --schedule 150 225 --wd 5e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/resnext-16x64d_$c --compression $c
done

for c in ${COMPRESSIONS[@]}; do
    nvprof -o results/profiling/wrn28-$c python profiling.py -a wrn --depth 28 --depth 28 --widen-factor 10 --drop 0.3 --epochs 200 --schedule 60 120 160 --wd 5e-4 --gamma 0.2 --checkpoint checkpoints/cifar10/WRN-28-10-drop_$c --compression $c
done

for c in ${COMPRESSIONS[@]}; do
    nvprof -o results/profiling/densenet100-$c python profiling.py -a densenet --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/densenet-bc-100-12_$c --compression $c
done

for c in ${COMPRESSIONS[@]}; do
    nvprof -o results/profiling/densenet190-$c python profiling.py -a densenet --depth 190 --growthRate 40 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/densenet-bc-L190-k40_$c --compression $c
done
