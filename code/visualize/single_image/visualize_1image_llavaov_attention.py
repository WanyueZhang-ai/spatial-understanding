import os

# 设置CUDA可见设备
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
import warnings
base_dir = "YOUR_PATH"
sys.path.append(os.path.join(base_dir, "LLaVA-NeXT"))
sys.path.append(os.path.join(base_dir, "whatsup_vlms"))
# sys.path.append(os.path.join(base_dir, "Qwen2.5-VL"))
# sys.path.append(os.path.join(base_dir, "Qwen2.5-VL/qwen-vl-utils/src"))
# print(sys.path)

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
import matplotlib.pyplot as plt
# from qwen_vl_utils import process_vision_info
from qwen_vl_utils.vision_process import process_vision_info
import math 
import matplotlib.image as mpimg

from PIL import Image
import requests
import copy
import torch

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

def calculate_plt_size(attention_layer_num):    
    num_layers = attention_layer_num
    cols = math.ceil(math.sqrt(num_layers))  
    rows = math.ceil(num_layers / cols)     
    return rows, cols

pretrained = "YOUR_PATH/llava-onevision-qwen2-7b-ov"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()

image_path = 'YOUR_PATH/liancang.jpg'
# question = 'What is the spatial relationship between the door in the picture and the artwork in the picture?'
# question = 'Where is the door in the picture?'
question = 'What is on the left side of the door in the picture?'
save_path = './output_llavaov/llavaov_attention_heatmaps_door_left.png'

image = Image.open(image_path)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
question = DEFAULT_IMAGE_TOKEN + question + "Answer the question using a single word or phrase."
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]

general_question = DEFAULT_IMAGE_TOKEN + "Write a general description of the image. Answer the question using a single word or phrase."
general_conv = copy.deepcopy(conv_templates[conv_template])
general_conv.append_message(general_conv.roles[0], general_question)
general_conv.append_message(general_conv.roles[1], None)
general_prompt_question = general_conv.get_prompt()

general_input_ids = tokenizer_image_token(general_prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]

output_shape = 0

with torch.no_grad():
    pos = 0
    pos_end = 0

    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        output_attentions=True
    )
    output = tokenizer.batch_decode(cont, skip_special_tokens=True)
    print(output)
    
    general_cont = model.generate(
        general_input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        output_attentions=True
    )
    general_output = tokenizer.batch_decode(general_cont, skip_special_tokens=True)
    print(general_output)
    

    # rows, cols = calculate_plt_size(len(output.attentions))
    # fig, axes = plt.subplots(rows, cols, figsize=(10.8, 16))
    # for i, ax in enumerate(axes.flatten()):
    #     if i < len(output.attentions): 
    #         att = output.attentions[i][0, :, -1, pos:pos_end].mean(dim=0)
    #         att = att.to(torch.float32).detach().cpu().numpy()

    #         general_att = general_output.attentions[i][0, :, -1, pos:pos_end].mean(dim=0)
    #         general_att = general_att.to(torch.float32).detach().cpu().numpy()

    #         att = att / general_att

    #         ax.imshow(att.reshape(output_shape), cmap="viridis", interpolation="nearest")
    #         ax.set_title(f"Layer {i+1}")
    #         ax.axis("off")
    #     else:
    #         ax.axis("off")

    original_img = mpimg.imread(image_path)
    # +1 给原图预留一个subplot
    total_plots = len(output.attentions) + 1
    rows, cols = calculate_plt_size(total_plots)

    fig, axes = plt.subplots(rows, cols, figsize=(10.8, 16))
    # fig, axes = plt.subplots(rows, cols, figsize=(10.8, 16), constrained_layout=True)

    for i, ax in enumerate(axes.flatten()):
        if i < len(output.attentions):
            att = output.attentions[i][0, :, -1, pos:pos_end].mean(dim=0)
            att = att.to(torch.float32).detach().cpu().numpy()

            general_att = general_output.attentions[i][0, :, -1, pos:pos_end].mean(dim=0)
            general_att = general_att.to(torch.float32).detach().cpu().numpy()

            att = att / general_att

            ax.imshow(att.reshape(output_shape), cmap="viridis", interpolation="nearest")
            ax.set_title(f"Layer {i+1}")
            ax.axis("off")
        elif i == total_plots - 1:
            ax.imshow(original_img)
            ax.set_title("Original Image")
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()    
    # fig.suptitle(question, fontsize=14, y=1.02)
    text = f'Question: {question}'
    fig.text(0.5, 0.01, text, ha='center', fontsize=12)
    # plt.show()    
    plt.savefig(save_path) 
    print("✅ ")