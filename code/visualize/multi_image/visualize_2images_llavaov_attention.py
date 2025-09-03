import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
import warnings
warnings.filterwarnings("ignore")

base_dir = "YOUR_PATH"
sys.path.extend([
    os.path.join(base_dir, "LLaVA-NeXT"),
    os.path.join(base_dir, "whatsup_vlms"),
    os.path.join(base_dir, "Qwen2.5-VL"),
    os.path.join(base_dir, "Qwen2.5-VL/qwen-vl-utils/src")
])

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from qwen_vl_utils.vision_process import process_vision_info

from PIL import Image
import requests
import copy
import torch

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

# === Step 1: Load model ===
pretrained = "YOUR_PATH"
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

# === Step 2: Input two images ===
image_paths = [ \
    'YOUR_PATH/50_76_9/one_50_76_rgb_image.png', \
    'YOUR_PATH/50_76_9/two_76_9_rgb_image.png' \
    ]

images = [Image.open(path) for path in image_paths]
image_tensors = process_images(images, image_processor, model.config)
image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors]
conv_template = "qwen_1_5"
question = f"{DEFAULT_IMAGE_TOKEN} {DEFAULT_IMAGE_TOKEN} From the perspective of the first picture, where is the bicycle on the carpet?"

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size for image in images]


# === Step 3: Image preprocessing ===
output_shapes = 0

# === Step 4: Token construction ===

general_question = f"{DEFAULT_IMAGE_TOKEN} {DEFAULT_IMAGE_TOKEN} Describe the images in general."
general_conv = copy.deepcopy(conv_templates[conv_template])
general_conv.append_message(general_conv.roles[0], question)
general_conv.append_message(general_conv.roles[1], None)
general_prompt_question = general_conv.get_prompt()

general_input_ids = tokenizer_image_token(general_prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size for image in images]

# === Step 5: Run model forward ===
with torch.no_grad():
    # 获取两张图的位置
    start_pos = []
    end_pos = []

    cont = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        output_attentions=True
    )
    output = tokenizer.batch_decode(cont, skip_special_tokens=True)
    print(output[0])
    
    general_cont = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        output_attentions=True
    )
    general_output = tokenizer.batch_decode(general_cont, skip_special_tokens=True)
    print(general_output[0])

# === Step 6: Visualize each image separately ===
def calculate_plt_size(layer_count):
    cols = math.ceil(math.sqrt(layer_count))
    rows = math.ceil(layer_count / cols)
    return rows, cols

def visualize_attention(img_index, save_path):
    start = start_pos[img_index]
    end = end_pos[img_index]
    original_img = mpimg.imread(image_paths[img_index])
    h, w = output_shapes[img_index]

    total_plots = len(output.attentions) + 1
    rows, cols = calculate_plt_size(total_plots)
    fig, axes = plt.subplots(rows, cols, figsize=(10.8, 16))

    for i, ax in enumerate(axes.flatten()):
        if i < len(output.attentions):
            att = output.attentions[i][0, :, -1, start:end].mean(dim=0)
            general_att = general_output.attentions[i][0, :, -1, start:end].mean(dim=0)

            att = att.to(torch.float32).detach().cpu().numpy()
            general_att = general_att.to(torch.float32).detach().cpu().numpy() + 1e-6
            norm_att = att / general_att

            ax.imshow(norm_att.reshape(h, w), cmap="viridis", interpolation="nearest")
            ax.set_title(f"Layer {i+1}")
            ax.axis("off")
        elif i == total_plots - 1:
            ax.imshow(original_img)
            ax.set_title("Original Image")
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    fig.text(0.5, 0.01, f'Question: {question}', ha='center', fontsize=12)
    plt.savefig(save_path)
    print(f"✅ : {save_path}")

os.makedirs('./output_llavaov/', exist_ok=True)
visualize_attention(0, './output_llavaov/llavaov_attention_image0.png')
visualize_attention(1, './output_llavaov/llavaov_attention_image1.png')
