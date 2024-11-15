# gist-cllm

## Generate Trajectories

Here we don't use aug.
```
python -m data.generate_trajectory \
    --max_new_tokens 16 \
    --max_new_seq_len 1024 \
    --use_labels \
    --data_size 10000 \
    --model models/Abel-7B-001 \
    --filename data/raw_data/gsm8k_train.jsonl
```

## Train

1. Generate Trajectories or directly download Jacobi trajectory to `data/collected_jacobi_trajectory/` from [HuggingFace](https://huggingface.co/datasets/xxxbrem/jacobi_trajectory_abel1)

2. Download target model [Abel-7B-001](https://huggingface.co/GAIR/Abel-7B-001)

3. Training:
`bash scripts/train_cllm.sh {model_path} {trajectory_file} {output_path} {n_token_seq_size}`

e.g.:
```bash
bash scripts/train_cllm.sh models/Abel-7B-001 data/collected_jacobi_trajectory/cleaned_gsm8k_train.jsonl_jacobi_max_new_tokens16_augFalse_labels_True_max_seq_len_1024_gist.json out 64
```
