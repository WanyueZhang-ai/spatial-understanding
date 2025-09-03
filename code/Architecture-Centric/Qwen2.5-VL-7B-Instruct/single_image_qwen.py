
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

from model_zoo import get_model
from dataset_zoo import get_dataset
from misc import seed_all, _default_collate, save_scores

from PIL import Image
import requests
import copy
from tqdm import tqdm

import matplotlib.pyplot as plt
import cv2
import numpy as np

# python main_aro.py --dataset=$dataset --model-name=$model_name
def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--model-name", default="Qwen2.5-VL-7B-Instruct", type=str, \
            choices=["openai-clip:ViT-B/32", "openai-clip:ViT-L/14", \
                "NegCLIP", "laion-clip:roberta-ViT-B/32", \
                "coca", "xvlm-pretrained-4m", "xvlm-pretrained-16m", \
                "blip-base-14m", "blip-base-129m", "flava", \
                "coca-cap", "xvlm-flickr", "xvlm-coco", \
                "blip-flickr-base", "blip-coco-base"])
    parser.add_argument("--dataset", default="VG_Relation", type=str, \
            choices=["VG_Relation", "VG_Attribution", "COCO_Order", \
            "Flickr30k_Order", "Controlled_Images_A", "Controlled_Images_B", \
            "COCO_QA_one_obj", "COCO_QA_two_obj", "VG_QA_one_obj", "VG_QA_two_obj"])
    parser.add_argument("--seed", default=1, type=int)
    
    parser.add_argument("--download", action="store_true", help="Whether to download the dataset if it doesn't exist. (Default: False)")
    parser.add_argument("--save-scores", action="store_true", help="Whether to save the scores for the retrieval to analyze later.")
    parser.add_argument("--output-dir", default="./outputs/outputs_qwen2.5_vl/", type=str)
    return parser.parse_args()

def load_model(dataset):
# def load_model(image, caption_options):
    warnings.filterwarnings("ignore")
    device = 'cuda'
    model_path = 'YOUR_PATH'
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, padding_side='left', use_fast=True)
    # model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    # model.eval()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            ).eval().to(device)
    
    correct_num = 0

    output_jsonl_path = os.path.join(args.output_dir, f"{args.dataset}_output_qwen2.5vl.jsonl")

    for idx, item in enumerate(tqdm(dataset)):
        # if idx==5:
        #     break
        image_path = item.image_options[0] 
        caption_options = item.caption_options
        labels = ["A", "B", "C", "D"]

        caption_text = "\n".join([f"{labels[i]}: {opt}" for i, opt in enumerate(caption_options)])
        question = (
            "Based on the image, choose the correct option from the list below. Only respond with the corresponding letter (e.g., A).\n\n"
            # "Based on the image, choose the correct option from the list below. Only respond with the corresponding letter (e.g., C).\n\n"
            f"{caption_text}"
        )

        messages_query = [
            {
                "role": "user",
                "content": [
                    {"type": "image","image": image_path,"max_pixels": 512*28*28},
                    {"type": "text", "text": question},
                ],
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

        print(response_text)

        output_entry = {
            "index": idx,
            "question": question,
            "output": response_text
        }
        with open(output_jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(output_entry, ensure_ascii=False) + "\n")

        if evaluate_scores(response_text):
            correct_num += 1

    accuracy_mean = correct_num / (len(dataset)+1)
    return accuracy_mean
    

def evaluate_scores(output):
    correct_pred = {
        "VG_Relation": 1,
        "VG_Attribution": 1,
        "COCO_Order": 0,
        "Flickr30k_Order": 0,
        "Controlled_Images_A": 0,
        "Controlled_Images_B": 0,
        "COCO_QA_one_obj": 0,
        "COCO_QA_two_obj": 0,
        "VG_QA_one_obj": 0,
        "VG_QA_two_obj": 0,
    }

    labels = ["A", "B", "C", "D"]
    correct_label = labels[correct_pred[args.dataset]]

    if correct_label in output:
        return True
    else:
        return False
    
def main(args):
    seed_all(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    # model, image_preprocess = get_model(args.model_name, args.device)
    image_preprocess = None
    
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=args.download)

    accuracy_mean = load_model(dataset)

    # correct_num = 0
    # for item in tqdm(dataset):
    #     image = item.image_options[0]
    #     caption_options = item.caption_options
    #     output = load_model(image, caption_options)
    #     if evaluate_scores(output):
    #         correct_num += 1
    
    # accuracy_mean = correct_num / (len(dataset)+1)
    jsonl_output_file = os.path.join(args.output_dir, "all_accuracy_qwen2.5_vl.jsonl")
    accuracy_data = {
        "Model": args.model_name,
        "Mean_Accuracy": accuracy_mean,
        "Dataset": args.dataset, 
        "Seed": args.seed
        }
    if os.path.exists(jsonl_output_file):
        with open(jsonl_output_file, 'a') as file:
            json.dump(accuracy_data, file)
            file.write('\n')
    else:
        with open(jsonl_output_file, 'w') as file:
            json.dump(accuracy_data, file)
            file.write('\n')
    print(accuracy_data)

    
if __name__ == "__main__":
    args = config()
    main(args)
