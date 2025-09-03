
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
# sys.path.append(os.path.join(base_dir, "LLaVA-NeXT"))
sys.path.append(os.path.join(base_dir, "whatsup_vlms"))
# sys.path.append(os.path.join(base_dir, "Qwen2.5-VL"))
# sys.path.append(os.path.join(base_dir, "Qwen2.5-VL/qwen-vl-utils/src"))
# print(sys.path)

import torch
import matplotlib.pyplot as plt
from PIL import Image
# from qwen_vl_utils import process_vision_info
import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from modelscope import AutoModel, AutoTokenizer
# from transformers import AutoConfig

from torch.utils.data import DataLoader

from dataset_zoo import get_dataset
from misc import seed_all, _default_collate, save_scores

from PIL import Image
import requests
import copy
from tqdm import tqdm

import matplotlib.pyplot as plt
import cv2
import numpy as np

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     print(e)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    # print(image_file)
    # image = Image.open(image_file).convert('RGB')
    image = image_file
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# python main_aro.py --dataset=$dataset --model-name=$model_name
def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--model-name", default="Mono-InternVL-2B", type=str, \
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
    parser.add_argument("--output-dir", default="./outputs/outputs_mono/", type=str)
    return parser.parse_args()

def load_model(dataset):
# def load_model(image, caption_options):
    warnings.filterwarnings("ignore")
    path = 'YOUR_PATH/Mono-InternVL-2B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    
    model.eval()
    
    correct_num = 0
    
    import os
    output_jsonl_path = os.path.join(args.output_dir, f"{args.dataset}_output_mono.jsonl")

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
        
        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=1024, do_sample=True)
        
        # single-image single-round conversation (单图单轮对话)
        question = '<image>\n' + question
        os.environ['NUM_IMAGES'] = str(1)
        
        response_text = model.chat(tokenizer, pixel_values, question, generation_config)
        print(response_text)
        
        # 保存输出
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
    jsonl_output_file = os.path.join(args.output_dir, "all_accuracy_mono.jsonl")
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
