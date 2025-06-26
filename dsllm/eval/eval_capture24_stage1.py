import os
import json
import re
from transformers import AutoTokenizer, AutoConfig
from dsllm.model.chronos_model import *
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dsllm.model import *
from dsllm.data.stage1_dataset import UniChannelTimeSeriesDataset
from dsllm.data.utils import generate_chat_template
from dsllm.utils import disable_torch_init
import warnings
import argparse
import time
import numpy as np


warnings.filterwarnings("ignore")

SYS_INST = """Analyze the time-series sensor data and describe the trends. For each time segment, state whether the trend is rising, falling, or constant.

Example 1:
Question: Analyze the sensor data trend from 0.0 to 0.1 seconds.
Response:
Time segments:
0.00-0.03s: rising
0.03-0.07s: falling
0.07-0.10s: constant

Counts:
Rising segments: 1
Falling segments: 1
Constant segments: 1

Summary: The data shows alternating patterns with a brief rising trend, followed by a longer falling trend, and ending in a constant phase.

Example 2:
Question: What are the trend changes from 0.0 to 0.15 seconds?
Response:
Time segments:
0.00-0.05s: constant
0.05-0.10s: rising
0.10-0.15s: falling

Counts:
Rising segments: 1
Falling segments: 1
Constant segments: 1

Summary: The data begins with a steady phase, transitions to a rising trend, and concludes with a falling pattern.

Now analyze the given sensor data following this format."""


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model_name_or_path', type=str,
                        default="")
    parser.add_argument('--pt_encoder_backbone_ckpt', type=str,
                        default="")
    parser.add_argument('--tokenize_method', type=str, default="MeanScaleUniformBins")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])

    parser.add_argument('--dataset', type=str, default="usc-had")
    parser.add_argument('--output_file_name', type=str, default="eval.json")
    parser.add_argument('--model_max_length', type=int, default=8192, help='context length during evaluation')
    parser.add_argument('--data_path', type=str, default="",
                        help="Path to the testing data.")
    parser.add_argument('--qa_path', type=str, default="",
                        help="Path to the testing QA data.")
    parser.add_argument('--ignore_qa_types', type=str, nargs='*', default=["sub_trend_no_val"])
    parser.add_argument('--debug', action='store_true', help='Run evaluation on first 10 samples only')

    # * data loader, batch_size, shuffle, num_workers
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()
    return args


def load_dataset(data_path, qa_path, chronos_tokenizer):
    print("Loading validation datasets.")
    dataset = UniChannelTimeSeriesDataset(
        data_path=data_path,
        qa_path=qa_path,
        tokenizer=None,  # * load ts and QA
        chronos_tokenizer=chronos_tokenizer,
        data_args=args
    )
    print(f"Example data: {dataset[5]}")
    print("Done!")
    print(dataset)
    return dataset


def custom_collate_fn(batch):
    batch_dict = {
        'question': [],
        'ground_truth': [],
        'type': [],
        'ts_token_ids': [],
        'ts_attention_mask': []
    }

    for item in batch:
        for key in batch_dict:
            batch_dict[key].append(item[key])

    return batch_dict


def get_dataloader(dataset, batch_size, num_workers=0):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=custom_collate_fn)
    return dataloader


def init_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name_or_path)

    # * print the model_name (get the basename)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left"
    )
    model = SensorLLMStage1LlamaForCausalLM.from_pretrained(
        model_name, 
        low_cpu_mem_usage=True,
        use_cache=True,
        torch_dtype=args.torch_dtype,
        device_map="auto"
    )
    model.get_model().load_pt_encoder_backbone_checkpoint(args.pt_encoder_backbone_ckpt,
                                                          tc=args.tokenize_method,
                                                          torch_dtype=args.torch_dtype)
    pt_backbone_config = AutoConfig.from_pretrained(args.pt_encoder_backbone_ckpt)

    assert hasattr(pt_backbone_config, "chronos_config"), "Not a Chronos config file"

    chronos_config = ChronosConfig(**pt_backbone_config.chronos_config)
    chronos_config.tokenizer_class = args.tokenize_method
    chronos_tokenizer = chronos_config.create_tokenizer()

    model.initialize_tokenizer_ts_backbone_config_wo_embedding(tokenizer, dataset=args.dataset)
    model.get_model().load_start_end_tokens(dataset=args.dataset)

    return model, tokenizer, chronos_tokenizer


def validate_sensor_data(ts_token_ids, ts_attention_mask):
    """Validate sensor data and provide detailed diagnostics"""
    diagnostics = []
    is_valid = True
    
    for i, (tokens, mask) in enumerate(zip(ts_token_ids, ts_attention_mask)):
        # Remove padding and special tokens
        valid_mask = mask[0] == 1  # Only consider tokens that aren't padding
        valid_tokens = tokens[0][valid_mask]
        valid_tokens = valid_tokens[valid_tokens < 2000]  # Remove special tokens
        
        # Basic statistics
        if len(valid_tokens) == 0:
            diagnostics.append(f"Stream {i}: No valid sensor readings found")
            is_valid = False
            continue
            
        stats = {
            "min": valid_tokens.min().item(),
            "max": valid_tokens.max().item(),
            "mean": valid_tokens.float().mean().item(),
            "std": valid_tokens.float().std().item(),
            "zeros": (valid_tokens == 0).sum().item(),
            "total": len(valid_tokens)
        }
        
        # Check for potential issues
        if stats["total"] < 10:  # Arbitrary minimum length
            diagnostics.append(f"Stream {i}: Too few readings ({stats['total']})")
            is_valid = False
        
        if stats["std"] == 0:
            diagnostics.append(f"Stream {i}: No variation in readings (constant value)")
            is_valid = False
            
        # Report on zero values but don't invalidate
        if stats["zeros"] > 0:
            zero_percentage = (stats["zeros"] / stats["total"]) * 100
            diagnostics.append(f"Stream {i}: Contains {zero_percentage:.1f}% zero values")
            
        diagnostics.append(
            f"Stream {i} stats: {stats['total']} readings, "
            f"range [{stats['min']:.2f}, {stats['max']:.2f}], "
            f"mean {stats['mean']:.2f}, std {stats['std']:.2f}"
        )
    
    return is_valid, diagnostics


def generate_outputs(model, tokenizer, inputs, ts_token_ids, ts_attention_mask, do_sample=True, temperature=0.7,
                     top_k=0, max_length=512, top_p=0.9, repetition_penalty=1.3, length_penalty=1.2):
    model.eval()
    model.get_model().pt_encoder_backbone.eval()
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    print(f"\nValidating sensor data...")
    is_valid, diagnostics = validate_sensor_data(ts_token_ids, ts_attention_mask)
    for msg in diagnostics:
        print(msg)
    if not is_valid:
        print("WARNING: Potential issues found with sensor data")
    
    print(f"\nStarting inference with parameters:")
    print(f"Temperature: {temperature}")
    print(f"Top-k: {top_k} (disabled)")
    print(f"Top-p: {top_p}")
    print(f"Repetition penalty: {repetition_penalty}")
    print(f"Length penalty: {length_penalty}")
    print(f"Max length: {max_length}")
    
    print(f"Input shapes:")
    print(f"- Input IDs: {inputs.input_ids.shape}")
    print(f"- Attention mask: {inputs.attention_mask.shape}")
    print(f"- TS token IDs: {[ts.shape for ts in ts_token_ids]}")
    print(f"- TS attention mask: {[mask.shape for mask in ts_attention_mask]}")
    
    start_time = time.time()
    try:
        with torch.inference_mode(), torch.cuda.amp.autocast():
            outputs = model.generate(
                **inputs,
                ts_token_ids=ts_token_ids,
                ts_attention_mask=ts_attention_mask,
                do_sample=do_sample,
                use_cache=False,
                temperature=temperature,
                top_k=top_k,
                max_new_tokens=max_length,
                min_new_tokens=50,  # Ensure some minimum output
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                no_repeat_ngram_size=4,  # Prevent repeating 4-grams
                eos_token_id=terminators,
                pad_token_id=tokenizer.pad_token_id,
                max_time=60.0,
                early_stopping=True,
                num_beams=1,  # Simple greedy decoding
                bad_words_ids=[[tokenizer.convert_tokens_to_ids(t)] for t in ["<", ">", "assistant", "human", "system"]]  # Prevent generating chat markers
            )
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
        raise e
    
    end_time = time.time()
    print(f"Inference took {end_time - start_time:.2f} seconds")
    print(f"GPU Memory after generation: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    
    input_token_len = inputs.input_ids.shape[1]
    n_diff_input_output = (inputs.input_ids != outputs[:, :input_token_len]).sum().item()

    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(outputs[:, input_token_len:], skip_special_tokens=True)
    outputs = [output.strip() for output in outputs]
    
    # Add output analysis
    for i, output in enumerate(outputs):
        print(f"\nOutput {i} analysis:")
        print(f"- Length: {len(output.split())} words")
        print(f"- First 50 chars: {output[:50]}...")
        
        # Check for repetition
        words = output.split()
        unique_words = len(set(words))
        if len(words) > 0:
            diversity_ratio = unique_words / len(words)
            print(f"- Word diversity ratio: {diversity_ratio:.2f}")
            if diversity_ratio < 0.3:  # Arbitrary threshold
                print("WARNING: Low word diversity detected")
    
    return outputs


def start_generation(model, tokenizer, dataloader, output_dir, output_file_name):
    results = {"prompt": SYS_INST}
    responses = []

    o_i = 0
    max_samples = 10 if args.debug else float('inf')  # Limit samples in debug mode
    print(f"\n{'DEBUG MODE: ' if args.debug else ''}Processing {'first 10' if args.debug else 'all'} samples...")
    
    for batch in tqdm(dataloader):
        if o_i >= max_samples:
            print("\nDebug mode: Reached 10 sample limit")
            break
            
        ts_token_ids = [ts_tensor.cuda() for ts_tensor in batch["ts_token_ids"]]
        ts_attention_mask = [ts_tensor.cuda() for ts_tensor in batch["ts_attention_mask"]]

        ground_truths = batch["ground_truth"]
        types = batch["type"]
        questions = batch["question"]

        print("\n" + "="*50)
        print(f"Processing sample {o_i}")
        print(f"Question type: {types[0]}")
        print(f"Raw question (with <ts> tokens): {questions[0]}")
        
        # Print actual sensor data values
        print("\nActual Sensor Data Values:")
        for i, ts_tensor in enumerate(ts_token_ids):
            print(f"\nSensor stream {i}:")
            # Remove padding tokens (token ID 1) and special tokens
            valid_tokens = ts_tensor[0][ts_tensor[0] != 1]
            valid_tokens = valid_tokens[valid_tokens < 2000]  # Assuming special tokens are > 2000
            print(f"Number of actual sensor readings: {len(valid_tokens)}")
            print(f"Value range: [{valid_tokens.min().item():.2f}, {valid_tokens.max().item():.2f}]")
            print(f"Mean: {valid_tokens.float().mean().item():.2f}")
            print(f"First 10 sensor values: {valid_tokens[:10].cpu().numpy()}")
            
            # Print distribution of values
            hist = torch.histc(valid_tokens.float(), bins=5)
            print("\nValue distribution (5 bins):")
            for bin_idx, count in enumerate(hist):
                print(f"Bin {bin_idx}: {count.item():.0f} values")
        print("="*50 + "\n")

        templated_questions = [generate_chat_template([
            {"role": "system", "content": SYS_INST},
            {"role": "user", "content": q}], bos_token=tokenizer.bos_token, eos_token=tokenizer.eos_token,
            add_generation_prompt=True) for q in
            questions]

        inputs = tokenizer(templated_questions, padding=True, return_tensors="pt").to(model.device)
        outputs = generate_outputs(model, tokenizer, inputs, ts_token_ids,
                                   ts_attention_mask)

        # saving results
        for q, gt, output, tp, ts in zip(questions, ground_truths, outputs, types, ts_token_ids):
            responses.append({
                "questions": q,
                "ground_truth": gt,
                "model_output": output,
                "model_len": len(ts[0]),
                "type": tp
            })
            print("\nOutput:", output)
            print("\nGround truth:", gt)
            print("-"*100)
        o_i += 1
    
    results["results"] = responses
    results["debug_mode"] = args.debug  # Add flag to results to indicate debug mode
    
    evaluate_generation(results, output_dir, output_file_name)


def evaluate_generation(results, output_dir, output_file_name):

    os.makedirs(output_dir, exist_ok=True)
    # save the results to a JSON file
    with open(os.path.join(output_dir, output_file_name), 'w') as fp:
        json.dump(results, fp, indent=2)

    # * print info
    print(f"Saved results to {os.path.join(output_dir, output_file_name)}")

    return results



def eval(args):
    # * ouptut
    args.output_dir = os.path.join(args.model_name_or_path, "evaluation")
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
