import os
import json
import re
from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizer
from dsllm.model.chronos_model import *
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dsllm.model import *
from dsllm.data.stage2_dataset import MultiChannelTimeSeriesCLSDatasetStage2
from dsllm.data.utils import generate_chat_template
from dsllm.utils import disable_torch_init
import warnings
import argparse
import yaml
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

SYS_INST = "A chat between a curious human and an AI assistant. The assistant is given a sequence of N features that represent information extracted from sensor (time-series) readings. The original readings consisted of N data points collected at a sample rate of 100Hz. The assistant's task is to analyze the trends and patterns in the sensor readings by leveraging the encoded information within the features to answer the following specific questions provided by the human."


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model_name_or_path', type=str,
                        default="/home/willidragon/william-research/sensorllm/sensorllm/outputs/SensorLLM_train_stage2/capture24_stage2")
    parser.add_argument('--pt_encoder_backbone_ckpt', type=str,
                        default="/home/willidragon/william-research/sensorllm/sensorllm/capture24_imp/chronos/chronos-t5-large")
    parser.add_argument('--tokenize_method', type=str, default="MeanScaleUniformBins")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])

    parser.add_argument('--dataset', type=str, default="capture24")
    parser.add_argument('--output_file_name', type=str, default="eval_capture24.json")
    parser.add_argument('--model_max_length', type=int, default=8192, help='context length during evaluation')
    parser.add_argument('--data_path', type=str, 
                        default="/home/willidragon/william-research/sensorllm/sensorllm/data/stage_2/test/capture24_test_data_stage2.pkl",
                        help="Path to the testing data.")
    parser.add_argument('--qa_path', type=str, 
                        default="/home/willidragon/william-research/sensorllm/sensorllm/data/stage_2/test/capture24_test_qa_stage2_cls.json",
                        help="Path to the testing QA data.")
    parser.add_argument('--ignore_qa_types', type=str, nargs='*', default=["sub_trend_no_val"])

    # Dataset specific arguments
    parser.add_argument('--preprocess_type', type=str, default="stage2_cls")
    parser.add_argument('--preprocess_type_eval', type=str, default="stage2_cls")
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--add_ts_special_token_text', type=bool, default=False)

    # * data loader, batch_size, shuffle, num_workers
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    # Add ts_backbone_config from yaml file
    with open("/home/willidragon/william-research/sensorllm/sensorllm/model/ts_backbone.yaml", 'r') as f:
        args.ts_backbone_config = yaml.safe_load(f)

    return args


def load_dataset(data_path, qa_path, chronos_tokenizer):
    print("Loading validation datasets.")
    
    # Label mapping for capture24 dataset
    label2id = {
        "sleep": 0,
        "sitting": 1,
        "standing": 2,
        "walking": 3,
        "bicycling": 4,
        "vehicle": 5,
        "household-chores": 6,
        "manual-work": 7,
        "sports": 8,
        "mixed-activity": 9
    }
    
    dataset = MultiChannelTimeSeriesCLSDatasetStage2(
        data_path=data_path,
        qa_path=qa_path,
        tokenizer=None,  # * load ts and QA
        chronos_tokenizer=chronos_tokenizer,
        split="eval",
        label2id=label2id,
        data_args=args
    )
    print(f"Example data: {dataset[5]}")
    print("Done!")
    print(dataset)
    return dataset


def custom_collate_fn(batch):
    batch_dict = {
        'question': [],
        'answer': [],
        'type': [],
        'mts_token_ids': [],
        'mts_attention_mask': [],
        'mts_tokenizer_state': [],
        'added_str': [],
        'smry': [],
        'trend_text': [],
        'corr_text': []
    }

    for item in batch:
        batch_dict['question'].append(item['question'])
        batch_dict['answer'].append(item['answer'])
        batch_dict['type'].append('classification')  # For stage2 classification
        batch_dict['mts_token_ids'].append(item['mts_token_ids'])
        batch_dict['mts_attention_mask'].append(item['mts_attention_mask'])
        batch_dict['mts_tokenizer_state'].append(item['mts_tokenizer_state'])
        batch_dict['added_str'].append(item.get('added_str', ''))
        batch_dict['smry'].append(item.get('smry', ''))
        batch_dict['trend_text'].append(item.get('trend_text', ''))
        batch_dict['corr_text'].append(item.get('corr_text', ''))

    return batch_dict


def get_dataloader(dataset, batch_size, num_workers=2):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=custom_collate_fn)
    return dataloader


def init_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name_or_path)

    # * print the model_name (get the basename)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    # Load config first to determine model type
    config = AutoConfig.from_pretrained(model_name)
    
    # Load tokenizer using AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left"
    )

    # Load the appropriate model class based on config
    if config.model_type == "sensorllmstage2":
        from dsllm.model.stage2_sensorllm import SensorLLMStage2LlamaForSequenceClassification
        model = SensorLLMStage2LlamaForSequenceClassification.from_pretrained(
            model_name, 
            low_cpu_mem_usage=False, 
            use_cache=True,
            torch_dtype=args.torch_dtype
        ).cuda()
    else:
        model = SensorLLMStage1V2LlamaForCausalLM.from_pretrained(
            model_name, 
            low_cpu_mem_usage=False, 
            use_cache=True,
            torch_dtype=args.torch_dtype
        ).cuda()

    model.get_model().load_pt_encoder_backbone_checkpoint(
        args.pt_encoder_backbone_ckpt,
        tc=args.tokenize_method,
        torch_dtype=args.torch_dtype
    )
    
    pt_backbone_config = AutoConfig.from_pretrained(args.pt_encoder_backbone_ckpt)
    assert hasattr(pt_backbone_config, "chronos_config"), "Not a Chronos config file"

    chronos_config = ChronosConfig(**pt_backbone_config.chronos_config)
    chronos_config.tokenizer_class = args.tokenize_method
    chronos_tokenizer = chronos_config.create_tokenizer()

    model.initialize_tokenizer_ts_backbone_config_wo_embedding(tokenizer, dataset=args.dataset)
    model.get_model().load_start_end_tokens(dataset=args.dataset)

    return model, tokenizer, chronos_tokenizer


def generate_outputs(model, tokenizer, inputs, ts_token_ids, ts_attention_mask):
    model.eval()
    model.get_model().pt_encoder_backbone.eval()
    
    # Stack the list of tensors into a single batch tensor
    mts_token_ids = torch.stack(ts_token_ids)
    mts_attention_mask = torch.stack(ts_attention_mask)
    
    with torch.inference_mode():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            mts_token_ids=mts_token_ids,
            mts_attention_mask=mts_attention_mask,
            use_cache=False  # Disable caching since we're doing classification
        )  
        
        # Get predicted class indices
        predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Convert predictions to labels using the reverse label mapping
        id2label = {
            0: "sleep",
            1: "sitting",
            2: "standing",
            3: "walking",
            4: "bicycling",
            5: "vehicle",
            6: "household-chores",
            7: "manual-work",
            8: "sports",
            9: "mixed-activity"
        }
        
        outputs = [id2label[pred.item()] for pred in predictions]
    
    return outputs


def start_generation(model, tokenizer, dataloader, output_dir, output_file_name):
    model.eval()
    results = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Get input tensors
            mts_token_ids = [ts_tensor.cuda() for ts_tensor in batch["mts_token_ids"]]
            mts_attention_mask = [ts_tensor.cuda() for ts_tensor in batch["mts_attention_mask"]]
            mts_tokenizer_state = [ts_tensor.cuda() if isinstance(ts_tensor, torch.Tensor) else torch.tensor(ts_tensor).cuda() for ts_tensor in batch["mts_tokenizer_state"]]
            questions = batch["question"]
            answers = batch["answer"]
            types = batch["type"]
            
            # Get the actual data from the batch
            added_str = batch.get("added_str", [])
            smry = batch.get("smry", [])
            trend_text = batch.get("trend_text", [])
            corr_text = batch.get("corr_text", [])

            ground_truths = answers  # * list of string

            # For classification, we still need to tokenize the questions but don't need generation template
            inputs = tokenizer(questions, padding=True, return_tensors="pt").to(model.device)
            outputs = generate_outputs(model, tokenizer, inputs, mts_token_ids,
                                       mts_attention_mask)  # List of str, length is B

            # saving results and collecting predictions for metrics
            for idx, (q, gt, output, tp) in enumerate(zip(questions, ground_truths, outputs, types)):
                result = {
                    "questions": q,
                    "ground_truth": gt,
                    "model_output": output,
                    "type": tp,
                    "data_segment": added_str[idx] if idx < len(added_str) else "N/A",
                    "statistics": smry[idx] if idx < len(smry) else "N/A",
                    "trend_analysis": trend_text[idx] if idx < len(trend_text) else "N/A",
                    "correlation_analysis": corr_text[idx] if idx < len(corr_text) else "N/A"
                }
                results.append(result)
                all_preds.append(output)
                all_labels.append(gt)

    # Calculate metrics
    metrics = calculate_metrics(all_preds, all_labels, output_dir)
    
    # Combine results and metrics
    final_results = {
        "prompt": SYS_INST,
        "results": results,
        "evaluation_metrics": metrics
    }
    
    evaluate_generation(final_results, output_dir, output_file_name)
    return final_results


def calculate_metrics(predictions, ground_truth, output_dir):
    """Calculate comprehensive evaluation metrics."""
    # Get unique labels
    unique_labels = sorted(list(set(ground_truth)))
    
    # Calculate basic metrics
    clf_report = classification_report(ground_truth, predictions, labels=unique_labels, output_dict=True)
    conf_matrix = confusion_matrix(ground_truth, predictions, labels=unique_labels)
    
    # Calculate per-class accuracy
    per_class_acc = {}
    for label in unique_labels:
        mask = np.array(ground_truth) == label
        correct = np.array(predictions)[mask] == label
        per_class_acc[label] = correct.mean() if len(correct) > 0 else 0
    
    # Create confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    conf_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(conf_matrix_path)
    plt.close()
    
    # Prepare metrics dictionary
    metrics = {
        "overall_accuracy": accuracy_score(ground_truth, predictions),
        "per_class_metrics": clf_report,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": conf_matrix.tolist(),
        "confusion_matrix_plot": conf_matrix_path,
        "total_samples": len(ground_truth),
        "class_distribution": {label: ground_truth.count(label) for label in unique_labels},
        "error_analysis": {
            "misclassified_count": sum(1 for p, t in zip(predictions, ground_truth) if p != t),
            "most_common_errors": get_most_common_errors(predictions, ground_truth)
        }
    }
    
    return metrics


def get_most_common_errors(predictions, ground_truth):
    """Analyze most common misclassification patterns."""
    error_patterns = []
    for pred, true in zip(predictions, ground_truth):
        if pred != true:
            error_patterns.append((true, pred))
    
    error_counts = {}
    for true, pred in error_patterns:
        key = f"{true} → {pred}"
        error_counts[key] = error_counts.get(key, 0) + 1
    
    # Sort by frequency and get top 10
    sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_errors[:10])


def evaluate_generation(results, output_dir, output_file_name):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the full results to a JSON file
    with open(os.path.join(output_dir, output_file_name), 'w') as fp:
        json.dump(results, fp, indent=2)
    
    # Print summary of results
    metrics = results["evaluation_metrics"]
    print("\n=== Evaluation Summary ===")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print("\nPer-class Performance:")
    for class_name, class_metrics in metrics["per_class_metrics"].items():
        if isinstance(class_metrics, dict):  # Skip averages
            print(f"\n{class_name}:")
            print(f"  Precision: {class_metrics['precision']:.4f}")
            print(f"  Recall: {class_metrics['recall']:.4f}")
            print(f"  F1-score: {class_metrics['f1-score']:.4f}")
            print(f"  Support: {class_metrics['support']}")
    
    print("\nMost Common Misclassifications:")
    for error_pattern, count in metrics["error_analysis"]["most_common_errors"].items():
        print(f"  {error_pattern}: {count}")
    
    print(f"\nDetailed results saved to {os.path.join(output_dir, output_file_name)}")
    print(f"Confusion matrix plot saved as {metrics['confusion_matrix_plot']}")
    
    return results


def eval(args):
    # * output
    args.output_dir = os.path.join(args.model_name_or_path, "evaluation")
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    output_file_path = os.path.join(args.output_dir, args.output_file_name)

    if not os.path.exists(output_file_path):
        # * need inferencing
        model, tokenizer, chronos_tokenizer = init_model(args)
        ts_backbone_config = model.get_model().ts_backbone_config
        args.ts_backbone_config = ts_backbone_config

        dataset = load_dataset(args.data_path, args.qa_path, chronos_tokenizer)
        dataloader = get_dataloader(dataset, args.batch_size, args.num_workers)

        print(f'[INFO] Start generating results for {args.output_file_name}.')
        results = start_generation(model, tokenizer, dataloader, args.output_dir, args.output_file_name)

        del model
        del tokenizer
        torch.cuda.empty_cache()
    else:
        # * directly load the results
        print(f'[INFO] {output_file_path} already exists, directly loading...')
        with open(output_file_path, 'r') as fp:
            results = json.load(fp)
        print(results["results"][:10])


if __name__ == "__main__":
    args = parse_config()
    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    args.torch_dtype = dtype_mapping[args.torch_dtype]

    eval(args)
