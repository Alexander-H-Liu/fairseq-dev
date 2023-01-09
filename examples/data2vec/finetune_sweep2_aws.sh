#!/usr/bin/env zsh

n_gpu_total=8
n_cpu=6
update_freq='[1]'
port=12347

# Env (python env identical to data2vec)
fairseq_root="/data/home/wnhsu/fairseq_repos/fairseq-py-d2vq"
data_root="/fsx-wav2vec/abaevski/data/libri/10h/wav2vec/raw" # Path to data tsv files (e.g., train 10hr)
lm_path="/fsx-wav2vec/wnhsu/datasets/librispeech/lm_4gram/4-gram.bin" # I'm usomg official 4-gram.bin 
lexicon_path="/fsx-wav2vec/wnhsu/datasets/librispeech/lm_4gram/lexicon_ltr.lst" # I'm using official librispeech_lexicon.lst
pretrain_log_root="/fsx-wav2vec/wnhsu/projects/data2vec_quant/exps"

# Finetuning options
split="train" # finetuning {split}.tsv
valid_subset="dev_other" # tsv file name for validation
config_path=$fairseq_root/examples/wav2vec/config/finetuning/ # might wanna try hubert config in the furture
config=base_10h # depending on {split}
load_teacher_model=( "true" "false" "false" ) # true: finetune teacher model; false: student
finetune_ckpts=( "checkpoint_214_200000.pt" "checkpoint_321_300000.pt" "checkpoint_428_400000.pt" )


for i in {1..3}; do
    load_tm=${load_teacher_model[$i]}
    ft_ckpt=${finetune_ckpts[$i]}

    for run in $pretrain_log_root/pretrain_400k/*; do
        pretrain_ckpt_path="${run}/ckpt/${ft_ckpt}"

        # wait until ckpt exist
        until [ -f $pretrain_ckpt_path ]
        do
            sleep 300
        done

        log_dir="${run}/finetune/10h/${ft_ckpt}_teacher${load_tm}"

        mkdir -p $log_dir
        PYTHONPATH=$(pwd):$(pwd)/examples LOG_DIR=${log_dir} \
        python fairseq_cli/hydra_train.py -m \
        --config-dir $config_path \
        --config-name $config \
        hydra/launcher=submitit_slurm +run=slurm_1_aws $dep_str \
        common.log_file=$log_dir/log.txt \
        common.tensorboard_logdir=$log_dir/tb \
        common.user_dir=$fairseq_root/examples/data2vec \
        checkpoint.save_dir=$log_dir/ckpt \
        checkpoint.save_interval=5 dataset.validate_interval=5 \
        dataset.train_subset=$split \
        dataset.valid_subset=$valid_subset \
        dataset.num_workers=$n_cpu \
        model.w2v_path=$pretrain_ckpt_path \
        +model.load_teacher_model=$load_tm \
        distributed_training.distributed_world_size=$n_gpu_total \
        distributed_training.distributed_port=$port \
        +optimization.update_freq=$update_freq \
        task.data=$data_root \
        task.normalize=true \
        +criterion.wer_lexicon=$lexicon_path \
        +criterion.wer_kenlm_model=$lm_path \
        +criterion.wer_lm_weight=2.0 \
        +criterion.wer_word_score=-1.0 &
    done
done
