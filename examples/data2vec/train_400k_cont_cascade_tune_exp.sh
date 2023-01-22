#!/usr/bin/env zsh

# Setting used in MIT (default 32GB GPU x16)
n_gpu_total=16
n_cpu=6
update_freq='[1]' # Avoid using update_freq>1 since codebook is updated per forward pass
port=12347

# Env (python env identical to data2vec)
fairseq_root="/private/home/wnhsu/data2vec_quant/fairseq-dev"
data_root="/checkpoint/wnhsu/data/hubert/manifests/pretrain/data/ls960/" # Path to data tsv files (libri 960hr)
train_subset="train"
valid_subset="dev_other" # tsv file name for validation

# Log
log_interval=200
save_every_n_updates=50000 # hopefully % 200k == 0
keep_interval_updates=-1 # keep ckpts if space permit 

log_dir=$1
freeze_pre_enc_module=$2
cfg_postfix=$3

PYTHONPATH=$(pwd):$(pwd)/examples LOG_DIR=${log_dir} \
python fairseq_cli/hydra_train.py -m \
--config-dir $fairseq_root/examples/data2vec/config/audio/pretraining \
--config-name base_discrete_400k${cfg_postfix} \
hydra/launcher=depedency_submitit_slurm +run=slurm_2 \
+next_script=${fairseq_root}/examples/data2vec/finetune_dep_400k.sh \
task.data=$data_root \
dataset.train_subset=$train_subset \
dataset.valid_subset=$valid_subset \
dataset.num_workers=$n_cpu \
common.log_interval=$log_interval \
common.log_file=$log_dir/log.txt \
common.tensorboard_logdir=$log_dir/tb \
common.user_dir=$fairseq_root/examples/data2vec \
checkpoint.save_dir=$log_dir/ckpt \
checkpoint.save_interval_updates=$save_every_n_updates \
checkpoint.keep_interval_updates=$keep_interval_updates \
distributed_training.distributed_world_size=$n_gpu_total \
distributed_training.distributed_port=$port \
model.codebook_size=256 \
model.normal_init_codebook=false \
model.average_top_k_layers=8 \
model.mask_prob=0.65 \
model.codebook_init_decay=0.9 \
model.freeze_pre_enc_modules=$freeze_pre_enc_module \
+lr_scheduler.final_lr_scale=0.1 \
+optimization.update_freq=$update_freq &
