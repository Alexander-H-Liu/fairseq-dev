#!/usr/bin/env zsh

# Default setting, changing one hyper-param at a time
normal_init_codebook=true
codebook_init_decay=0.6 # 0.9?

# sweep codebook size
codebook_sizes=( "350" "400" "450" )
top_ks=( "8" "10" )
mask_probs=( "0.65" "0.70" "0.75" "0.80" )

for codebook_size in ${codebook_sizes[@]};do
    for top_k in ${top_ks[@]};do
        for mask_prob in ${mask_probs[@]};do
            zsh examples/data2vec/train_400k_cascade_tune.sh $codebook_size $top_k $normal_init_codebook $mask_prob $codebook_init_decay
        done
    done
done