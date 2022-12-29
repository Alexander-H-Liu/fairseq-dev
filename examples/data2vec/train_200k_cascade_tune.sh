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
log_root="/checkpoint/wnhsu/projects/data2vec_quant/exps"
log_interval=200
save_every_n_updates=20000 # hopefully % 200k == 0
keep_interval_updates=-1 # keep ckpts if space permit 

# Hyper-param. (prioritized in order)
### Size of codebook
codebook_size=$1  # 512>1024>2048>256>128
### Targeting top K layers, each with a codebook
top_k=$2 # 8>10>6>12>4>2 (not sure if 10 & 12 is going to run OOM)
### Codebook init. method
normal_init_codebook="$3" # false: random then Instance norm, true: normal(0,1/|codebook|**2)
### Masking prob
mask_prob="$4"      # worth tuning?
### TBD
freeze_codebook_step=200000 # see if training is unstable before 200k
codebook_init_decay="$5" # so far so good (unstable is caused by LR or teacher EMA, not codebook decay)

log_dir=$log_root/pretrain/base_discrete_v${codebook_size}_norm${normal_init_codebook}_k${top_k}_m${mask_prob}_cbdec${codebook_init_decay}
mkdir -p $log_dir

set -x
PYTHONPATH=$(pwd):$(pwd)/examples LOG_DIR=${log_dir} \
python fairseq_cli/hydra_train.py -m \
--config-dir $fairseq_root/examples/data2vec/config/audio/pretraining \
--config-name base_discrete_200k \
hydra/launcher=depedency_submitit_slurm +run=slurm_2 \
+next_script=${fairseq_root}/examples/data2vec/finetune_dep.sh \
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
model.codebook_size=$codebook_size \
model.normal_init_codebook=$normal_init_codebook \
model.average_top_k_layers=$top_k \
model.mask_prob=$mask_prob \
model.freeze_codebook_step=$freeze_codebook_step \
model.codebook_init_decay=$codebook_init_decay \
+optimization.update_freq=$update_freq &
