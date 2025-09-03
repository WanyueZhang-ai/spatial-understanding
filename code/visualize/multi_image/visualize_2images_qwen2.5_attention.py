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

# === Step 1: Load model ===
device = 'cuda'
model_path = 'YOUR_PATH/Qwen2.5-VL-7B-Instruct'

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
).eval().to(device)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, padding_side='left', use_fast=True)

# === Step 2: Input two images ===
image_paths = ['YOUR_PATH/attention_visualize_cot/images/data-v3_sc3_staging_16-v2/11_14/14_rgb_image.png', 'YOUR_PATH/attention_visualize_cot/images/data-v3_sc3_staging_16-v2/11_14/11_rgb_image.png']
question = "Which object is closest to the bike in both images? \nChoose the correct option from the list below. Only respond with the corresponding letter.\n\nA. sofa\nB. blue mat\nC. indoor_plant\nD. Other\n"

messages_query = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image_paths[0], "max_pixels": 512*28*28},
        {"type": "image", "image": image_paths[1], "max_pixels": 512*28*28},
        {"type": "text", "text": f"{question}"},
    ],
}]

# === Step 3: Image preprocessing ===
image_inputs, _ = process_vision_info(messages_query)
image_inputs_aux = processor.image_processor(images=image_inputs)
image_grid_thw = image_inputs_aux["image_grid_thw"].numpy()  # shape: (2, 3)

output_shapes = (image_grid_thw[:, 1:] // 2).astype(int)  # (h, w) for each image

# === Step 4: Token construction ===
text_query = processor.apply_chat_template(messages_query, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text_query], images=image_inputs, padding=True, return_tensors="pt").to(device)

# Also get general description to normalize attention
messages_general = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image_paths[0], "max_pixels": 512*28*28},
        {"type": "image", "image": image_paths[1], "max_pixels": 512*28*28},
        {"type": "text", "text": "Describe the images in general."},
    ],
}]
text_general = processor.apply_chat_template(messages_general, tokenize=False, add_generation_prompt=True)
general_inputs = processor(text=[text_general], images=image_inputs, padding=True, return_tensors="pt").to(device)

# === Step 5: Run model forward ===
with torch.no_grad():
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    input_ids = inputs['input_ids'].tolist()[0]

    start_pos = []
    end_pos = []
    for _ in range(2):
        start = input_ids.index(vision_start_token_id) + 1
        end = input_ids.index(vision_end_token_id)
        start_pos.append(start)
        end_pos.append(end)
        input_ids = input_ids[end + 1:]

    # import pdb; pdb.set_trace()
    output = model(**inputs, output_attentions=True)
    general_output = model(**general_inputs, output_attentions=True)

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

    # plt.tight_layout()
    # fig.text(0.5, 0.01, f'Question: {question}', ha='center', fontsize=12)

    title_text = f'Question: {question}'.replace('\n', '\\n')
    fig.text(0.5, 0.01, title_text, ha='center', fontsize=12, wrap=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(save_path)
    print(f"✅ : {save_path}")

def visualize_attention_layers_separately(img_index, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    start = start_pos[img_index]
    end = end_pos[img_index]
    original_img = mpimg.imread(image_paths[img_index])
    h, w = output_shapes[img_index]

    # 保存原图
    plt.figure()
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "original_image.png"))
    plt.close()

    for i, att_layer in enumerate(output.attentions):
        att = att_layer[0, :, -1, start:end].mean(dim=0)
        general_att = general_output.attentions[i][0, :, -1, start:end].mean(dim=0)

        att = att.to(torch.float32).detach().cpu().numpy()
        general_att = general_att.to(torch.float32).detach().cpu().numpy() + 1e-6
        norm_att = att / general_att

        plt.figure()
        plt.imshow(norm_att.reshape(h, w), cmap="viridis", interpolation="nearest")
        plt.title(f"Layer {i+1}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"layer_{i+1}.png"))
        plt.close()

        print(f"✅ : {os.path.join(save_dir, f'layer_{i+1}.png')}")

os.makedirs('./output_qwen2.5_VL_7b-3/', exist_ok=True)
visualize_attention(0, './output_qwen2.5_VL_7b-3/qwen2.5_7b_attention_image0.png')
visualize_attention(1, './output_qwen2.5_VL_7b-3/qwen2.5_7b_attention_image1.png')

os.makedirs('./output_qwen2.5_VL_7b_layers-3/', exist_ok=True)
visualize_attention_layers_separately(0, './output_qwen2.5_VL_7b_layers-3/image0')
visualize_attention_layers_separately(1, './output_qwen2.5_VL_7b_layers-3/image1')
