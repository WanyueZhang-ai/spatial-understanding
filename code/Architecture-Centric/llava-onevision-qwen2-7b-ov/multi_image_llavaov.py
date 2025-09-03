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
# sys.path.append(os.path.join(base_dir, "Qwen2.5-VL"))
# sys.path.append(os.path.join(base_dir, "Qwen2.5-VL/qwen-vl-utils/src"))
# print(sys.path)
from transformers import AutoProcessor
import torch
import matplotlib.pyplot as plt
# from qwen_vl_utils import process_vision_info
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from torch.utils.data import DataLoader
from PIL import Image
import requests
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np
import re

import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

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
    parser.add_argument("--model-name", default="llava-onevision-qwen2-7b-ov", type=str)
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
    
    if output == correct_answer:
        return True
    
    pattern = rf'^{correct_answer}([^a-zA-Z]|$)'
    match = re.match(pattern, output)
    if match:
        remaining_part = output[match.end():].strip()
        if re.search(r'[A-Da-d]', remaining_part):
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
    warnings.filterwarnings("ignore")
    pretrained = "YOUR_PATH/llava-onevision-qwen2-7b-ov"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    llava_model_args = {
            "multimodal": True,
        }
    overwrite_config = {}
    overwrite_config["image_aspect_ratio"] = "pad"
    llava_model_args["overwrite_config"] = overwrite_config
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)

    model.eval()
    
    output_jsonl_path = os.path.join(args.output_dir, f"{args.dataset}_output_llavaov.jsonl")
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

        images = [Image.open(image) for image in item.image_options]
        num_images = len(images)
        os.environ['NUM_IMAGES'] = str(num_images)
        
        image_tensors = process_images(images, image_processor, model.config)
        image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors]

        conv_template = "qwen_1_5"
        question = f"{DEFAULT_IMAGE_TOKEN}{DEFAULT_IMAGE_TOKEN}\n" + query

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [image.size for image in images]

        neg_200_pos = (input_ids == -200).nonzero(as_tuple=True)

        if neg_200_pos[0].numel() > 0:
            first_index = neg_200_pos[1][0].item()
            last_index = neg_200_pos[1][-1].item()

            before_count = first_index
            after_count = input_ids.size(1) - last_index - 1
            os.environ['SYS_PROMPT'] = str(before_count)
            os.environ['QUESTION'] = str(after_count)
        # Generate response
        cont = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        response_text = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        print(response_text)

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
    jsonl_output_file = os.path.join(args.output_dir, "all_accuracy_llavaov.jsonl")
    with open(jsonl_output_file, 'a') as f:
        f.write(json.dumps(accuracy_data, ensure_ascii=False) + "\n")

    print(accuracy_data)


if __name__ == "__main__":
    args = config()
    main(args)
