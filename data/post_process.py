import json

def clear_itr_0(input_path, output_path):
    with open(input_path, 'r') as file:
        data = json.load(file)
        data_clear_itr_0 = [d for d in data if d.get('jacobian_itr_id') != 'itr_0']

    with open(output_path, 'w') as outfile:
        json.dump(data_clear_itr_0, outfile)

def construct_gist_training_data(input_path, output_path):
    with open(input_path, 'r') as file:
        data = json.load(file)
    
    data_gist = []
    window_size = 4
    tmp = []
    for i in range(len(data)):
        if len(tmp) < window_size - 1:
            tmp.append(data[i])
            continue
        if data[i]['data_id'] != tmp[-1]['data_id']:
            current_data_id = tmp[-1]['data_id']
            if data_gist[-1]['teacher_output_ids'][0] != data_gist[-1]['complete_teacher_output_ids']:
                while data_gist[-1]['data_id'] == current_data_id:
                    data_gist.pop()
            tmp = [data[i]]
            continue
        dic = {}
        tmp.append(data[i])
        dic['data_id'] = data[i]['data_id']
        dic['jacobian_itr_id'] = tmp[0]['jacobian_itr_id']
        dic['prompt_ids_len'] = tmp[0]['prompt_ids_len']
        dic['prompt_ids'] = tmp[0]['prompt_ids']
        dic['answer_trajectory_ids'] = []
        
        answer_trajectory_ids_len = [len(tmp[j]['answer_trajectory_ids']) for j in range(window_size)]
        for j in range(max(answer_trajectory_ids_len)):
            added_trajectory = []
            for k in range(window_size):
                added_trajectory += tmp[k]['answer_trajectory_ids'][j] if j < len(tmp[k]['answer_trajectory_ids']) else tmp[k]['answer_trajectory_ids'][-1]
            dic['answer_trajectory_ids'].append(added_trajectory)

        dic['labels_ids'] = tmp[0]['labels_ids']
        dic['teacher_output_ids'] = [data_gist[-1]['teacher_output_ids'][0] + data[i]['answer_trajectory_ids'][-1]] if dic['jacobian_itr_id'] != 'itr_0' \
                                    else [dic['prompt_ids'][0][0] + dic['answer_trajectory_ids'][-1]]
        dic['complete_teacher_output_ids'] = tmp[0]['complete_teacher_output_ids']
        data_gist.append(dic)
        tmp = tmp[1:]

    with open(output_path, 'w') as outfile:
        json.dump(data_gist, outfile)


input_path = 'data/collected_jacobi_trajectory/cleaned_gsm8k_train.jsonl_jacobi_max_new_tokens16_augFalse_labels_True_max_seq_len_1024.json'
output_path = 'data/collected_jacobi_trajectory/cleaned_gsm8k_train.jsonl_jacobi_max_new_tokens16_augFalse_labels_True_max_seq_len_1024_gist.json'
construct_gist_training_data(input_path, output_path)