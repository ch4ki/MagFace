#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

la=10
ua=110
lm=0.45
um=0.8
lg=20

# settings
MODEL_ARC=iresnet100
OUTPUT=./test/

mkdir -p ${OUTPUT}/vis/

python -u trainer.py \
    --arch iresnet50 \
    --train_list /training/face-group/opensource/ms1m-112/ms1m_train.list \
    --workers 8 \
    --epochs 25 \
    --start-epoch 0 \
    --batch-size 512 \
    --embedding-size 512 \
    --last-fc-size 85742 \
    --arc-scale 64 \
    --learning-rate 0.1 \
    --momentum 0.9 \
    --weight-decay 5e-4 \
    --lr-drop-epoch 10 18 22 \
    --lr-drop-ratio 0.1 \
    --print-freq 100 \
    --pth-save-fold ${OUTPUT} \
    --pth-save-epoch 1 \
    --l_a ${la} \
    --u_a ${ua} \
    --l_margin ${lm} \
    --u_margin ${um} \
    --lambda_g ${lg} \
    --vis_mag 1       