import json

def clear_itr_0(input_path, output_path):
    with open(input_path, 'r') as file:
        data = json.load(file)
        data_clear_itr_0 = [d for d in data if d.get('jacobian_itr_id') != 'itr_0']

    with open(output_path, 'w') as outfile:
        json.dump(data_clear_itr_0, outfile)

input_path = 'data/collected_jacobi_trajectory/cleaned_Abel-7B-001_jacobi_max_new_tokens16_augTrue_labels_True_max_seq_len_1024.json'
output_path = 'data/collected_jacobi_trajectory/cleaned_Abel-7B-001_jacobi_max_new_tokens16_augTrue_labels_True_max_seq_len_1024_clear_itr_0.json'
clear_itr_0(input_path, output_path)