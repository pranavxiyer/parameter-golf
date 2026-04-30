# Notable Non-Record Submission: {} BPB — Learned Adapters on Random Linear Maps

## Summary

## Key Architecture Change
The main idea behind this submission is the idea of learned adapters on random linear maps. In the baseline, within each block's MLP, there are 2 large matrices stored &mdash; fc of (hidden, dim) and proj of (dim, hidden). Instead, the AdapterMLP utilizes a random seed to generate W_fc and W_proj of the same dimensions as fc and proj in the baseline. However, these do not take up space within the artifact, since they are randomly generated and not fixed parameters (random linear map). In order to actually have meaningful computation within the MLP, there are 2 low-rank matrices stored for each W_fc and W_proj, essentially acting as LoRA matrices that facilitate learning (learned adapters). The dimensions of A, B for W_fc are (hidden, rank) and (rank, dim). The dimensions of A, B for W_proj are (dim, rank) and (rank, hidden). 

Instead of 2 * hidden * dim parameters for the MLP within each transformer block, the AdapterMLP architecture requires 2 * (rank * hidden + rank * dim) parameters. This artifact reduction is what allowed me to expand the number of transformer layers and mlp_mult factor. 

Baseline MLP (mlp_mult = 2):
2 * 1024 * 512 = 1,048,576 parameters per block

Adapter MLP (rank = 160, mlp_mult = 3):
2 * ()

## Space Savings

## Changes from Baseline
1. 12 transformer layers
2. random adapter MLPs &mdash; which allowed for additional transformer layers
3. MLP mult 3x (instead of 2)
4. sequence length 2048
5. mixed int-6/int-8 compression

random adapter MLPs + mixed int-6/int-8 compression + sliding eval

## 10-minute wallclock

## 30-minute wallclock