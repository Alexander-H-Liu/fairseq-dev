#!/usr/bin/env zsh

# Default setting, changing one hyper-param at a time
codebook_size=256
top_k=8
normal_init_codebook=false
mask_prob=0.8
codebook_init_decay=0.9


# sweep codebook size
codebook_sizes=( "200" "400" "300" "150" )
for sweep_codebook_size in ${codebook_sizes[@]};do
    ./train_200k_cascade_tune.sh $sweep_codebook_size $top_k $normal_init_codebook $mask_prob $codebook_init_decay
done

# sweep top_k
top_ks=( "10" "12" "6" "4" )
for sweep_top_k in ${top_ks[@]};do
    ./train_200k_cascade_tune.sh $codebook_size $sweep_top_k $normal_init_codebook $mask_prob $codebook_init_decay
done

# sweep codebook init. 
sweep_normal_init_codebook=true
./train_200k_cascade_tune.sh $codebook_size $top_k $sweep_normal_init_codebook $mask_prob $codebook_init_decay

# sweep mask prob
mask_probs=( "0.75" "0.85" "0.90" )
for sweep_mask_prob in ${mask_probs[@]};do
    ./train_200k_cascade_tune.sh $codebook_size $top_k $normal_init_codebook $sweep_mask_prob $codebook_init_decay
done

# sweep codebook decay (quite different for first 80k steps, not sure afterward)
codebook_init_decays=( "0.8" "0.7" "0.6" "0.5" )
for sweep_codebook_init_decay in ${codebook_init_decays[@]};do
    ./train_200k_cascade_tune.sh $codebook_size $top_k $normal_init_codebook $sweep_mask_prob $sweep_codebook_init_decay
done

