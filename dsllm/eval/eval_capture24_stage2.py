import os
import json
import re
from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizer
from dsllm.model.chronos_model import *
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dsllm.model import *
from dsllm.data.stage2_dataset import MultiChannelTimeSeriesCLSDatasetStage2, DataCollatorForTsCLSDatasetStage2
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
from dsllm.model.stage2_sensorllm import SensorLLMStage2LlamaForSequenceClassification
from dataclasses import dataclass


warnings.filterwarnings("ignore")

SYS_INST = "A chat between a curious human and an AI assistant. The assistant is given a sequence of N features that represent information extracted from sensor (time-series) readings. The original readings consisted of N data points collected at a sample rate of 100Hz. The assistant's task is to analyze the trends and patterns in the sensor readings by leveraging the encoded information within the features to answer the following specific questions provided by the human."

# Load label2id and id2label from YAML config globally
with open("/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/model/ts_backbone.yaml", "r") as f:
    ts_config = yaml.safe_load(f)
    id2label = ts_config["capture24"]["id2label"]
    label2id = {v: k for k, v in id2label.items()}

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--pt_encoder_backbone_ckpt', type=str, required=True)
    parser.add_argument('--tokenize_method', type=str, default="MeanScaleUniformBins")
    parser.add_argument('--torch_dtype', type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--qa_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--output_file_name', type=str, default="eval_capture24.json")
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--preprocess_type', type=str, default="stage2_cls")
    parser.add_argument('--add_ts_special_token_text', type=bool, default=False)
    args = parser.parse_args()
    return args

@dataclass
class EvalDataCollatorForTsCLSDatasetStage2:
    tokenizer: PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels, mts_token_ids, mts_attention_mask, mts_tokenizer_state = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "mts_token_ids", "mts_attention_mask", "mts_tokenizer_state")
        )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        # Pass through input_texts and answer as lists
        input_texts = [instance.get("input_texts", "") for instance in instances]
        answers = [instance.get("answer", "") for instance in instances]

        return dict(
            input_ids=input_ids,
            labels=torch.tensor(labels),
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            mts_token_ids=torch.stack(mts_token_ids),
            mts_attention_mask=torch.stack(mts_attention_mask),
            mts_tokenizer_state=mts_tokenizer_state,
            input_texts=input_texts,
            answer=answers
        )

def main():
    args = parse_config()
    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    args.torch_dtype = dtype_mapping[args.torch_dtype]

    # Model
    disable_torch_init()
    model = SensorLLMStage2LlamaForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        torch_dtype=args.torch_dtype
    ).cuda()
    model.get_model().load_pt_encoder_backbone_checkpoint(
        args.pt_encoder_backbone_ckpt,
        tc=args.tokenize_method,
        torch_dtype=args.torch_dtype
    )
    # Ensure token IDs are set before loading start/end tokens
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
    model.initialize_tokenizer_ts_backbone_config_wo_embedding(tokenizer, dataset="capture24")
    model.get_model().load_start_end_tokens(dataset="capture24")
    model.eval()

    # Tokenizer
    pt_backbone_config = AutoConfig.from_pretrained(args.pt_encoder_backbone_ckpt)
    chronos_config = ChronosConfig(**pt_backbone_config.chronos_config)
    chronos_config.tokenizer_class = args.tokenize_method
    chronos_tokenizer = chronos_config.create_tokenizer()

    # Dataset
    class DummyArgs:
        pass
    data_args = DummyArgs()
    data_args.preprocess_type = args.preprocess_type
    data_args.shuffle = args.shuffle
    data_args.dataset = "capture24"
    data_args.ts_backbone_config = ts_config
    data_args.add_ts_special_token_text = args.add_ts_special_token_text

    dataset = MultiChannelTimeSeriesCLSDatasetStage2(
        data_path=args.data_path,
        qa_path=args.qa_path,
        tokenizer=tokenizer,
        chronos_tokenizer=chronos_tokenizer,
        split="eval",
        label2id=label2id,
        data_args=data_args
    )
    collator = EvalDataCollatorForTsCLSDatasetStage2(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator)

    all_preds = []
    all_labels = []
    results = []
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        mts_token_ids = batch["mts_token_ids"].cuda()
        mts_attention_mask = batch["mts_attention_mask"].cuda()
        mts_tokenizer_state = batch["mts_tokenizer_state"]
        labels = batch["labels"].cuda()
        # Get input question and answer if available
        input_texts = batch.get("input_texts", None)
        answers = batch.get("answer", None)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mts_token_ids=mts_token_ids,
                mts_attention_mask=mts_attention_mask,
                mts_tokenizer_state=mts_tokenizer_state
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            for i in range(len(preds)):
                result = {
                    "ground_truth": id2label[all_labels[-len(preds)+i]],
                    "ground_truth_idx": all_labels[-len(preds)+i],
                    "model_output": id2label[preds[i].item()],
                    "model_output_idx": preds[i].item()
                }
                # Add the exact model input string as 'input_text'
                if input_texts is not None:
                    result["input_text"] = input_texts[i]
                if answers is not None:
                    result["answer"] = answers[i]
                results.append(result)

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Get the unique classes present in the data
    unique_labels = sorted(set(all_labels + all_preds))
    target_names_filtered = [id2label[i] for i in unique_labels]
    
    clf_report = classification_report(all_labels, all_preds, labels=unique_labels, target_names=target_names_filtered, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=unique_labels)

    output = {
        "accuracy": accuracy,
        "classification_report": clf_report,
        "confusion_matrix": conf_matrix.tolist(),
        "results": results
    }
    if "_upsampled" in args.output_file_name:
        output_dir = os.path.join(args.model_name_or_path, "evaluation_upsampled")
    else:
        output_dir = os.path.join(args.model_name_or_path, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, args.output_file_name), "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved evaluation results to {os.path.join(output_dir, args.output_file_name)}")
    print(f"Accuracy: {accuracy:.4f}")

    # --- Plotting Section ---
    class_names = target_names_filtered  # Use only the classes present in the data

    # 1. Confusion Matrix Heatmap
    conf_matrix_np = np.array(conf_matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_np, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_heatmap.png"))
    plt.close()

    # 2. Classification Report Bar Plots
    report = clf_report
    precisions = [report[c]["precision"] for c in class_names]
    recalls = [report[c]["recall"] for c in class_names]
    f1s = [report[c]["f1-score"] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, precisions, width, label='Precision')
    plt.bar(x, recalls, width, label='Recall')
    plt.bar(x + width, f1s, width, label='F1-score')
    plt.xticks(x, class_names, rotation=45)
    plt.ylabel("Score")
    plt.title("Classification Report Metrics by Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "classification_report_metrics.png"))
    plt.close()

    # 3. Misclassification Distribution Bar Plots
    # Compute missclassification_report
    missclassification_report = {c: {} for c in class_names}
    for true_idx, pred_idx in zip(all_labels, all_preds):
        true_class = id2label[true_idx]
        pred_class = id2label[pred_idx]
        if true_class != pred_class:
            if pred_class not in missclassification_report[true_class]:
                missclassification_report[true_class][pred_class] = 0
            missclassification_report[true_class][pred_class] += 1
    # Save to output
    output["missclassification_report"] = missclassification_report
    with open(os.path.join(output_dir, args.output_file_name), "w") as f:
        json.dump(output, f, indent=2)
    # Plot
    for true_class in class_names:
        mis_dict = missclassification_report[true_class]
        if mis_dict:
            plt.figure(figsize=(8, 4))
            sns.barplot(x=list(mis_dict.keys()), y=list(mis_dict.values()))
            plt.title(f"Misclassification Distribution for '{true_class}'")
            plt.ylabel("Count")
            plt.xlabel("Predicted as")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"missclass_{true_class}.png"))
            plt.close()

    print("Graphs saved as PNG files in the evaluation directory.")

if __name__ == "__main__":
    main()
