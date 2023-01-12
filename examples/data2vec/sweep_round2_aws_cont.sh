#!/usr/bin/env zsh

# Continueing from 200k ckpt
log_root="/fsx-wav2vec/wnhsu/projects/data2vec_quant/exps"
resume_ckpt=$log_root/pretrain_400k/base_discrete_v350_normtrue_k10_m0.70_cbdec0.6/ckpt/checkpoint_214_200000.pt

freeze_teacher_steps=( "200000" "300000" ) # freeze teacher after 200k/300k updates
final_lr_scales=( "0.1" "0.2" ) # Final lr scale for tri-stage LR decay
freeze_pre_enc_modules=( "false" ) # freeze just transformer encoder or everything,  was "true"
mask_probs=( "0.70" "0.80" )

# Default setting, changing one hyper-param at a time
normal_init_codebook=true
codebook_init_decay=0.6 # 0.9?
codebook_size=350
top_k=8
for mask_prob in ${mask_probs[@]}; do
    for freeze_teacher_step in ${freeze_teacher_steps[@]}; do
        for final_lr_scale in ${final_lr_scales[@]};do
            for freeze_pre_enc_module in ${freeze_pre_enc_modules[@]};do
                # Copy 200k checkpoint to resume
                log_dir=$log_root/pretrain_400k/base_discrete_v${codebook_size}_norm${normal_init_codebook}_k${top_k}_m${mask_prob}_cbdec${codebook_init_decay}_flrs${final_lr_scale}_fpem${freeze_pre_enc_module}_freeze${freeze_teacher_step}
                mkdir -p $log_dir/ckpt/
                cp $resume_ckpt $log_dir/ckpt/checkpoint_last.pt
                zsh examples/data2vec/train_400k_aws_cont_cascade_tune.sh $codebook_size $top_k $normal_init_codebook $mask_prob $codebook_init_decay $final_lr_scale $freeze_pre_enc_module $freeze_teacher_step
            done
        done
    done
done
