# gist-cllm

## Train

1. Directly download Jacobi trajectory to `data/collected_jacobi_trajectory/` from [HuggingFace](https://huggingface.co/datasets/xxxbrem/jacobi_trajectory_abel1)

2. Download target model [Abel-7B-001](https://huggingface.co/GAIR/Abel-7B-001)

3. Training:
`bash scripts/train_cllm.sh {model_path} {trajectory_file} {output_path} {n_token_seq_size}`

e.g.:
```bash
bash scripts/train_cllm.sh models/Abel-7B-001 data/collected_jacobi_trajectory/cleaned_gsm8k_train.jsonl_jacobi_max_new_tokens64_augFalse_labels_True_max_seq_len_1024.json out 64
```
