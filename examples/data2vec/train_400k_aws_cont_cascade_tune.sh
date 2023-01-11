#!/usr/bin/env zsh

# Setting used in MIT (default 32GB GPU x16)
n_gpu_total=16
n_cpu=6
update_freq='[1]' # Avoid using update_freq>1 since codebook is updated per forward pass
port=12347

# Env (python env identical to data2vec)
fairseq_root="/data/home/wnhsu/fairseq_repos/fairseq-py-d2vq"
data_root="/fsx-wav2vec/abaevski/data/librispeech" # Path to data tsv files (libri 960hr)
train_subset="train"
valid_subset="dev_other" # tsv file name for validation

# Log
log_root="/fsx-wav2vec/wnhsu/projects/data2vec_quant/exps"
log_interval=200
save_every_n_updates=50000 # hopefully % 200k == 0
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
# freeze_codebook_step=200000 # see if training is unstable before 200k
codebook_init_decay="$5" # so far so good (unstable is caused by LR or teacher EMA, not codebook decay)


final_lr_scale=$6
freeze_pre_enc_module=$7
freeze_teacher_step=$8

log_dir=$log_root/pretrain_400k/base_discrete_v${codebook_size}_norm${normal_init_codebook}_k${top_k}_m${mask_prob}_cbdec${codebook_init_decay}_flrs${final_lr_scale}_fpem${freeze_pre_enc_module}_freeze${freeze_teacher_step}
mkdir -p $log_dir

PYTHONPATH=$(pwd):$(pwd)/examples LOG_DIR=${log_dir} \
python fairseq_cli/hydra_train.py -m \
--config-dir $fairseq_root/examples/data2vec/config/audio/pretraining \
--config-name base_discrete_400k_decay \
hydra/launcher=depedency_submitit_slurm +run=slurm_2_aws \
+next_script=${fairseq_root}/examples/data2vec/finetune_dep_400k_aws.sh \
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
model.codebook_init_decay=$codebook_init_decay \
model.freeze_codebook_step=$freeze_teacher_step \
model.freeze_pre_enc_modules=$freeze_pre_enc_module \
+lr_scheduler.final_lr_scale=$final_lr_scale \
+optimization.update_freq=$update_freq &
