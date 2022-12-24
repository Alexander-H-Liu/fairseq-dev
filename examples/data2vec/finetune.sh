#!/usr/bin/env zsh

n_gpu_total=8
n_cpu=6
update_freq='[1]'
port=12347

# Env (python env identical to data2vec)
fairseq_root=/path/to/fairseq-dev
log_root=/path/to/store/log # same as train_200k.sh
data_root=/path/to/data # Path to data tsv files (e.g., train 10hr)
lm_path=/path/to/lm.bin # I'm usomg official 4-gram.bin 
lexicon_path=/path/to/lexicon.lst # I'm using official librispeech_lexicon.lst

# Finetuning options
split=??? # finetuning {split}.tsv
valid_subset="dev-other" # tsv file name for validation
config_path=$fairseq_root/examples/wav2vec/config/finetuning/ # might wanna try hubert config in the furture
config=base_10h # depending on {split}
load_teacher_model=true # true: finetune teacher model; false: student

# Pretraining Hyper-param. (just for retrieving the ckpt)
codebook_size=512
top_k=8
normal_init_codebook=false
mask_prob=0.80


### Executable below
exp_name=base_discrete_v${codebook_size}_norm${normal_init_codebook}_k${top_k}_m${mask_prob}
pretrain_ckpt_path=$log_root/pretrain/$exp_name/ckpt/checkpoint_last.pt

log_dir=$log_root/finetune/$split/${exp_name}_teacher$load_teacher_model
mkdir -p $log_dir


fairseq-hydra-train \
--config-dir $fairseq_root/examples/wav2vec/config/finetuning \
--config-name base_10h \
common.log_file=$log_dir/log.txt \
common.tensorboard_logdir=$log_dir/tb \
common.user_dir=$fairseq_root/examples/data2vec \
checkpoint.save_dir=$log_dir/ckpt \
checkpoint.save_interval=5 dataset.validate_interval=5 \
dataset.train_subset=$split \
dataset.valid_subset=$valid_subset \
dataset.num_workers=$n_cpu \
model.w2v_path=$pretrain_ckpt_path \
+model.load_teacher_model=$load_teacher_model \
distributed_training.distributed_world_size=$n_gpu_total \
distributed_training.distributed_port=$port \
+optimization.update_freq=$update_freq \
task.data=$data_root \
task.normalize=true \
+criterion.wer_lexicon=$lexicon_path \
+criterion.wer_kenlm_model=$lm_path \
+criterion.wer_lm_weight=2.0 \
+criterion.wer_word_score=-1.0
