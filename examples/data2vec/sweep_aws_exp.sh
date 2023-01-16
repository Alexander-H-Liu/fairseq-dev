#!/usr/bin/env zsh

# Default setting, changing one hyper-param at a time
codebook_size=256
top_k=8
normal_init_codebook=false
mask_prob=0.8
codebook_init_decay=0.9


# sweep codebook size
codebook_sizes=( "64" "128" "1024" "2048" )
for sweep_codebook_size in ${codebook_sizes[@]};do
    zsh examples/data2vec/train_200k_aws_exp_cascade_tune.sh $sweep_codebook_size $top_k $normal_init_codebook $mask_prob $codebook_init_decay $codebook_init_decay
done

# sweep mask prob
mask_probs=( "0.70" "0.65" )
for sweep_mask_prob in ${mask_probs[@]};do
    zsh examples/data2vec/train_200k_aws_exp_cascade_tune.sh $codebook_size $top_k $normal_init_codebook $sweep_mask_prob $codebook_init_decay $codebook_init_decay
done

# sweep codebook decay (fixed value)
codebook_init_decays=( "0.999" "0.99" "0.8" "0.7" "0.6" "0.5" )
for sweep_codebook_init_decay in ${codebook_init_decays[@]};do
    zsh examples/data2vec/train_200k_aws_exp_cascade_tune.sh $codebook_size $top_k $normal_init_codebook $mask_prob  $sweep_codebook_init_decay $sweep_codebook_init_decay
done

# sweep codebook decay (decay)
codebook_end_decays=( "0.999" "0.99" )
for sweep_codebook_end_decay in ${codebook_end_decays[@]};do
    zsh examples/data2vec/train_200k_aws_exp_cascade_tune.sh $codebook_size $top_k $normal_init_codebook $mask_prob  $codebook_init_decay $sweep_codebook_end_decay
done





