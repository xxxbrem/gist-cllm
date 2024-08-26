import json

with open('/workspace/minghang/Consistency_LLM/data/collected_jacobi_trajectory/cleaned_l_jacobi_max_new_tokens16_augTrue_labels_True_max_seq_len_1024.json', 'r') as file:
    data = json.load(file)
    print()
    
    
with open("/workspace/minghang/Consistency_LLM/data/collected_jacobi_trajectory/data_processed.json", 'w') as f:
    json.dump(data, f)