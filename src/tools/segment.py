from pydantic import BaseModel, Field
from typing import Dict, Any, Union
from typing import Optional
import torch
import cv2
import os

import torch.nn.functional as F
import base64
from io import BytesIO
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from src.LISA.model.LISA import LISAForCausalLM
from src.LISA.model.llava import conversation as conversation_lib
from src.LISA.model.llava.mm_utils import tokenizer_image_token
from src.LISA.model.segment_anything.utils.transforms import ResizeLongestSide
from src.LISA.utils.utils import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX



def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def prompt_input(instruction):
    """
    instruction stands for question, and inputs are some prior knowledge you want to the add.
    """
    return (f"Below is a cell image segmentation task, paired with an input that provides further information."
            f"Output the segmentation mask precisely based on the given image and instruction.\n\n"
            f"### instruction:\n{instruction}\n\n### Answer:\n")




# Define the core segmentation function
def segment_image(prompt: str, image_path: str) -> str:
 
    # Prepare the parameters
    model_version = "/home/idies/workspace/Temporary/xyu1/scratch/LISA-13B-llama2-v1"  
    vision_tower_path = "/home/idies/workspace/Temporary/xyu1/scratch/hub/clip-vit-large-patch14"
    precision = "fp32"
    conv_type = "llava_v1"
    use_mm_start_end = True
    save_path = "/home/idies/workspace/Storage/xyu1/persistent/Langchain/ours_test/segment_resutls/results.png"

    
    # Initialize tokenizer and model
    print("Running LISA tool with parameters:")
    tokenizer = AutoTokenizer.from_pretrained(model_version, model_max_length=512, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    
    
    
    
    # Initialize model
    torch_dtype = torch.float32
    if precision == "bf16":
        torch_dtype = torch.bfloat16
    elif precision == "fp16":
        torch_dtype = torch.half

            
    model = LISAForCausalLM.from_pretrained(
        model_version, low_cpu_mem_usage=True, vision_tower=vision_tower_path, seg_token_idx=seg_token_idx
    )

    
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if precision == "bf16":
        model = model.bfloat16().cuda()

    elif precision == "fp32":
        model = model.float().cuda()

    #vision_tower.to(device=0)
    vision_tower.to("cuda")
    
    # Image processor and transformation
    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(1024)
    model.eval()
        
        
   
    
    
    
    
    # Preprocess image
    if not os.path.exists(image_path):
        print("File not found in {}".format(image_path))

    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    image_clip = (
        clip_image_processor.preprocess(image_np, return_tensors="pt")[
            "pixel_values"
        ][0]
        .unsqueeze(0)
        .cuda()
    )
    if precision == "bf16":
        image_clip = image_clip.bfloat16()
    elif precision == "fp16":
        image_clip = image_clip.half()
    else:
        image_clip = image_clip.float()

    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]

    image = (
        preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        .unsqueeze(0)
        .cuda()
    )
    if precision == "bf16":
        image = image.bfloat16()
    elif precision == "fp16":
        image = image.half()
    else:
        image = image.float()
        
        
    #prepare the prompt
    conv = conversation_lib.conv_templates[conv_type].copy()
    conv.messages = []
        
    
    prompt = prompt
    #prompt = prompt_input(query)
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    if use_mm_start_end:
        replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()
        


    # Evaluate model
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()
    

    # Get model predictions
    output_ids, pred_masks = model.evaluate(
        image_clip,
        image,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=512,
        tokenizer=tokenizer,
    ) 


     # Process results
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    
    
    for i, pred_mask in enumerate(pred_masks):
        if pred_mask.shape[0] == 0:
            continue

        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > 0


        save_img = image_np.copy()
        save_img[pred_mask] = (
            image_np * 0.5
            + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
        )[pred_mask]
        save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        
        image_save_path = save_path  # Original save path for the overlay image
        mask_save_path = save_path.replace('.png', '_mask.png')  # Modify as needed

        # Save the overlay image
        cv2.imwrite(image_save_path, save_img)

        # Prepare the binary mask for saving: convert Boolean to uint8 (0 or 255)
        mask_to_save = pred_mask.astype(np.uint8) * 255
        cv2.imwrite(mask_save_path, mask_to_save)
        
    #Plot the images
    # plt.figure(figsize=(15, 5))
    # plt.subplot(1, 3, 1)
    # plt.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    # plt.title('Original Image')
    # plt.axis('off')

    # # Prediction mask
    # plt.subplot(1, 3, 2)
    # plt.imshow(pred_mask, cmap='gray')
    # plt.title('Prediction Mask')
    # plt.axis('off')

    # # Masked image
    # plt.subplot(1, 3, 3)
    # plt.imshow(cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB))
    # plt.title('Masked Image')
    # plt.axis('off')

    # plt.show()

    
    
     # Convert image to Base64
    # buffered = BytesIO()
    # image = Image.fromarray(save_img)
    # image.save(buffered, format="PNG")
    # img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    #print("text_output: ", text_output)
    #return text_output, pred_masks, image_np
    #del model
    #del vision_tower, image_clip, image
    #torch.cuda.empty_cache()

    print(f"Segmentation completed and saved at {save_path}. mask saved at {mask_save_path}")
    
    return f"Segmentation completed successfully in segment_save_path:{save_path}, the corresponding mask saved in segment_mask_path:{mask_save_path}"


