import argparse
import os
import pandas as pd
import json
# 设置CUDA可见设备
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
import warnings
base_dir = "YOUR_PATH"
sys.path.append(os.path.join(base_dir, "LLaVA-NeXT"))
sys.path.append(os.path.join(base_dir, "whatsup_vlms"))
sys.path.append(os.path.join(base_dir, "Qwen2.5-VL"))
sys.path.append(os.path.join(base_dir, "Qwen2.5-VL/qwen-vl-utils/src"))
# print(sys.path)
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
import matplotlib.pyplot as plt
# from qwen_vl_utils import process_vision_info
from qwen_vl_utils.vision_process import process_vision_info

from torch.utils.data import DataLoader
from PIL import Image
import requests
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np
import re


# class CustomItem:
#     def __init__(self, image_1, image_2, caption_options, question, answer):
#         self.image_options = [image_1, image_2]
#         self.caption_options = caption_options
#         self.question = question
#         self.answer = answer

class CustomItem:
    def __init__(self, id, image_1, image_2, question, answer):
        self.id = id
        self.image_options = [image_1, image_2]
        self.question = question
        self.answer = answer



def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--model-name", default="Qwen2.5-VL-7B-Instruct", type=str)
    parser.add_argument("--dataset", default="Block_Restoration", type=str,
                        choices=["Distance_Assessment", "Block_Restoration", "Direction_Relationship_Transmission"])
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--output-dir", default="./outputs/outputs_qwen2.5_vl/", type=str)
    return parser.parse_args()


# def evaluate_scores(output, correct_answer):
#     return correct_answer.strip() in output

def evaluate_scores(output, correct_answer):
    output = output.strip()
    correct_answer = correct_answer.strip().upper()
    # import pdb; pdb.set_trace()
    if output.upper() == correct_answer:
        return True

    pattern = rf'^{correct_answer}([^a-zA-Z]|$)'
    match = re.match(pattern, output, flags=re.IGNORECASE)

    if match:
        remaining_part = output[match.end():].strip()

        other_options = "ABCD".replace(correct_answer, "")
        conflict_pattern = rf'\b[{other_options}]\b'

        conflict_match = re.search(conflict_pattern, remaining_part)#, flags=re.IGNORECASE)

        if conflict_match:
            return False
        return True

    return False

def load_jsonl_data(base_dir, filepath, dataset_type):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            obj = json.loads(line.strip())
            id = obj["id"]
            if dataset_type == "Distance_Assessment":
                image_1 = os.path.join(base_dir, obj["image_1"])
                image_2 = os.path.join(base_dir, obj["image_2"])
                # image_1 = obj["image_1"]
                # image_2 = obj["image_2"]
                # caption_options = [opt["text"] for opt in obj["options"]]
                question = obj["query"]
                answer = obj["answer"]
                # answer = obj["target_option"]
            elif dataset_type == "Block_Restoration":
                image_1 = os.path.join(base_dir, obj["image_1"])
                image_2 = os.path.join(base_dir, obj["mask_image"])
                # image_2 = obj["mask_image"]
                # caption_options = obj["options"]
                # question = obj["question"]
                question = obj["query"]
                answer = obj["answer"]
            elif dataset_type == "Direction_Relationship_Transmission":
                image_1 = os.path.join(base_dir, obj["image_1"])
                image_2 = os.path.join(base_dir, obj["image_2"])
                # caption_options = [opt["text"] for opt in obj["options"]]
                question = obj["query"]
                answer = obj["answer"]
            else:
                continue
            data.append(CustomItem(id, image_1, image_2, question, answer))
    return data


def load_model(args, dataset):
    device = args.device
    model_path = 'YOUR_PATH'
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, padding_side='left', use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval().to(device)

    output_jsonl_path = os.path.join(args.output_dir, f"{args.dataset}_output_qwen2.5vl.jsonl")
    correct_num = 0

    for idx, item in enumerate(tqdm(dataset)):
        # if idx ==500 :
        #     break
        # caption_options = item.caption_options
        labels = ["A", "B", "C", "D"]

        # caption_text = "\n".join([f"{labels[i]}: {opt}" for i, opt in enumerate(caption_options)])
        # question = item.question + f"\n\n{caption_text}"
        query = item.question
        # import pdb; pdb.set_trace()
        messages_query = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image, "max_pixels": 512*28*28}
                    for image in item.image_options
                ] + [{"type": "text", "text": query}],
            }
        ]


        image_inputs, _ = process_vision_info(messages_query)

        text_query = processor.apply_chat_template(
            messages_query,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(
            text=[text_query],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=200)
        response_text = processor.batch_decode(output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]

        correct = evaluate_scores(response_text, item.answer)
        if correct:
            correct_num += 1

        output_entry = {
            "id":  item.id,
            "query": query,
            "output": response_text,
            "ground_truth": item.answer,
            "correct": "True" if correct else "False"
        }
        with open(output_jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(output_entry, ensure_ascii=False) + "\n")

    accuracy = correct_num / len(dataset)
    return accuracy


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    base_dir = "YOUR_PATH"
    filename_map = {
        "Distance_Assessment": "merged_distance_questions.jsonl",
        "Block_Restoration": "merged_mask_questions.jsonl",
        "Direction_Relationship_Transmission": "merged_direction_questions.jsonl",
    }
    filepath = os.path.join(base_dir, filename_map[args.dataset])
    dataset = load_jsonl_data(base_dir, filepath, args.dataset)
    accuracy_mean = load_model(args, dataset)

    accuracy_data = {
        "Model": args.model_name,
        "Mean_Accuracy": accuracy_mean,
        "Dataset": args.dataset,
        "Seed": args.seed
    }
    jsonl_output_file = os.path.join(args.output_dir, "all_accuracy_qwen2.5_vl.jsonl")
    with open(jsonl_output_file, 'a') as f:
        f.write(json.dumps(accuracy_data, ensure_ascii=False) + "\n")

    print(accuracy_data)


if __name__ == "__main__":
    args = config()
    main(args)
