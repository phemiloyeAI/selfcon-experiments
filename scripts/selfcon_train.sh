#!/bin/bash

seed="0"
method="SelfCon"
data="flowers"
model="resnet18"
arch="resnet"
size="fc"
pos="[False,True,False]"
bsz="32" #1024
lr="0.005"  #0.5
label="True"
multiview="False" #False

python main_represent.py --exp_name "${arch}_${size}_${pos}" \
    --project_name "selfcon-resnet-results-reproduction" \
    --seed $seed \
    --method $method \
    --train_trainval \
    --model $model \
    --selfcon_pos $pos \
    --selfcon_arch $arch \
    --selfcon_size $size \
    --batch_size $bsz \
    --learning_rate $lr \
    --temp 0.1 \
    --epochs 1000 \
    --cosine \
    --precision \
    --data_cfg $1 


python main_linear.py --batch_size $bsz \
    --project_name "selfcon-resnet-results-reproduction" \
    --model $model \
    --learning_rate 3 \
    --weight_decay 0 \
    --selfcon_pos $pos \
    --selfcon_arch $arch \
    --selfcon_size $size \
    --epochs 100 \
    --lr_decay_epochs '60,80' \
    --lr_decay_rate 0.1 \
    --subnet \
    --data_cfg $1 \
    --train_trainval \
    --ckpt ./save/representation/${method}/${data}_models/${method}_${data}_${model}_lr_${lr}_multiview_${multiview}_label_${label}_decay_0.0001_bsz_${bsz}_temp_0.1_seed_${seed}_cosine_${arch}_${size}_${pos}/last.pth

