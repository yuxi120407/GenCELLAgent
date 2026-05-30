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


import os
import argparse

import torch
import numpy as np

from src.SegGPT.seggpt_engine import inference_image
import src.SegGPT.models_seggpt as models_seggpt


def prepare_model(chkpt_dir, arch='seggpt_vit_large_patch16_input896x448', seg_type='instance'):
    # build model
    model = getattr(models_seggpt, arch)()
    model.seg_type = seg_type
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model

# Define the core segmentation function
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def seggpt_inference_img(image_path: str, prompt_image_path: str, prompt_mask_path: str, save_dir: str = None) -> str:

    if not os.path.exists(prompt_image_path):
        error_message = f"Error: prompt_image_path does not exist: {prompt_image_path}"
        print(error_message)
        return error_message
    if not os.path.exists(prompt_mask_path):
        error_message = f"Error: prompt_mask_path does not exist: {prompt_mask_path}"
        print(error_message)
        return error_message

    device = torch.device('cuda')
    ckpt_path = os.path.join(_REPO_ROOT, "src", "SegGPT", "seggpt_vit_large.pth")
    model = prepare_model(ckpt_path).to(device)

    if save_dir is None:
        save_dir = os.path.join(_REPO_ROOT, "output", "seggpt_results")

    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    image_dir = os.path.join(save_dir, image_stem)
    os.makedirs(image_dir, exist_ok=True)

    save_path = os.path.join(image_dir, f"{image_stem}_seggpt.png")
    mask_path = os.path.join(image_dir, f"{image_stem}_seggpt_mask.png")
    inference_image(model, device, image_path, prompt_image_path, prompt_mask_path, save_path, mask_path)

    del model
    torch.cuda.empty_cache()

    print(f"One shot segmentation completed and saved at {save_path}, the mask is saved in seggpt_mask_path:{mask_path}.")
    return f"One shot segmentation completed successfully in seggpt_output_path:{save_path}, the mask is saved in seggpt_mask_path:{mask_path}"


