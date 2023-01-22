#!/usr/bin/env zsh

# Continueing from 200k ckpt
log_root="/fsx-wav2vec/wnhsu/projects/data2vec_quant/exps"
resume_ckpt=$log_root/pretrain/base_discrete_v256_normfalse_k8_m0.65_cbdec0.9_endcbdec0.9/ckpt/checkpoint_last.pt


freeze_pre_enc_modules=( "true" "false" )
config_postfix=( "_decay" "" )


for freeze_pre_enc_module in ${freeze_pre_enc_modules[@]}; do
    for cfg_postfix in ${config_postfix[@]}; do
        # Copy 200k checkpoint to resume
        log_dir=$log_root/pretrain/base_discrete_v256_normfalse_k8_m0.65_cbdec0.9_endcbdec0.9_freezecnn${freeze_pre_enc_module}${cfg_postfix}
        mkdir -p $log_dir/ckpt/
        cp $resume_ckpt $log_dir/ckpt/checkpoint_last.pt
        zsh examples/data2vec/train_400k_aws_cont_cascade_tune_exp.sh $log_dir $freeze_pre_enc_module $cfg_postfix
    done
done
