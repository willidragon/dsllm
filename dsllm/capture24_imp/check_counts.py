import pickle
import json

qa_path = "/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/data/stage_2_compare_buffer/300seconds_1000DS/train/capture24_train_qa_stage2_cls.json"
pkl_path = "/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/data/stage_2_compare_buffer/300seconds_1000DS/train/capture24_train_data_stage2_300seconds_1000DS.pkl"

with open(qa_path, "r") as f:
    qa = json.load(f)
qa_indices = [entry["index"] for entry in qa["dataset"]]
print("First 5 QA indices:", qa_indices[:5])
print("Last 5 QA indices:", qa_indices[-5:])
print("QA indices min:", min(qa_indices))
print("QA indices max:", max(qa_indices))
print("Actual QA indices count:", len(qa_indices))
print("Unique QA indices count:", len(set(qa_indices)))
missing = set(range(min(qa_indices), max(qa_indices)+1)) - set(qa_indices)
print("Missing QA indices:", sorted(missing))
duplicates = set([x for x in qa_indices if qa_indices.count(x) > 1])
print("Duplicate QA indices:", sorted(duplicates))

with open(pkl_path, "rb") as f:
    data = pickle.load(f)
print("First 5 PKL entry shapes:", [d.shape for d in data[:5]])
print("Last 5 PKL entry shapes:", [d.shape for d in data[-5:]])
print("PKL samples:", len(data)) 