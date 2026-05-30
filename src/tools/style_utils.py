# style_utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.autograd import Variable
import os
import tifffile as tiff
from skimage.exposure import equalize_adapthist, rescale_intensity

# --------- VGG MODEL ---------
class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        pool_layer = nn.MaxPool2d if pool == 'max' else nn.AvgPool2d
        self.pool1 = pool_layer(kernel_size=2, stride=2)
        self.pool2 = pool_layer(kernel_size=2, stride=2)
        self.pool3 = pool_layer(kernel_size=2, stride=2)
        self.pool4 = pool_layer(kernel_size=2, stride=2)
        self.pool5 = pool_layer(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]

# --------- GRAM MATRIX ---------
class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2)) 
        G.div_(h * w)
        return G

# --------- CORRELATION UTILS ---------
def layerwise_pearson_corr(f1, f2):
    a = f1.detach().cpu().numpy().flatten()
    b = f2.detach().cpu().numpy().flatten()
    return np.corrcoef(a, b)[0, 1]

def weighted_style_correlation(features_img1, features_img2, style_weights):
    assert len(features_img1) == len(features_img2) == len(style_weights), "Mismatch in layer count"
    correlations = [
        layerwise_pearson_corr(f1, f2)
        for f1, f2 in zip(features_img1, features_img2)
    ]
    weights = np.array(style_weights, dtype=np.float64)
    weights /= weights.sum()
    return float(np.sum(weights * np.array(correlations)))

# --------- PRE/POST PROCESSING ---------
img_size = 512
prep = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1,1,1]),
    transforms.Lambda(lambda x: x.mul_(255)),
])

postpa = transforms.Compose([
    transforms.Lambda(lambda x: x.mul_(1./255)),
    transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1,1,1]),
    transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
])
postpb = transforms.ToPILImage()

def postp(tensor):
    t = postpa(tensor)
    t[t > 1] = 1
    t[t < 0] = 0
    return postpb(t)

# --------- UTILS ---------
def load_image_with_norm(path: str, norm_type: str, dataset_name: str = None):
    """
    Load a single image and return a VGG-ready tensor.

    For Mouse_Brain: two-stage preprocessing
      1. Custom: 4× upscaling + 1-99th percentile normalization
      2. Then apply tool-specific norm (cellpose/microsam/cellsam)

    For other datasets: directly apply tool-specific norm
    """
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in ('.png', '.jpg', '.jpeg'):
            array = np.array(Image.open(path).convert('RGB'))
        else:
            array = tiff.imread(path)

        # Stage 1: Mouse_Brain custom preprocessing
        if dataset_name == "Mouse_Brain":
            array = _preprocess_mouse_brain(array)
            # Now array is uint8 RGB after upscaling + percentile norm

        # Stage 2: Apply tool-specific normalization
        rgb = _apply_norm(array, norm_type)
        img = Image.fromarray(rgb)
    except Exception as e:
        print(f"Could not read {path}: {e}")
        raise
    tensor = prep(img)
    if torch.cuda.is_available():
        return Variable(tensor.unsqueeze(0).cuda())
    return Variable(tensor.unsqueeze(0))


def load_vgg_model(model_path, use_cuda=True):
    model = VGG()
    model.load_state_dict(torch.load(model_path))
    for param in model.parameters():
        param.requires_grad = False
    if use_cuda and torch.cuda.is_available():
        model = model.cuda()
    return model

# def load_and_preprocess_images(image_dir, image_names):
#     imgs = [Image.open(f"{image_dir}{name}") for name in image_names]
#     imgs_torch = [prep(img) for img in imgs]
#     if torch.cuda.is_available():
#         imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
#     else:
#         imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
#     return imgs_torch


def _to_channel_last(array: np.ndarray) -> np.ndarray:
    """Ensure channel-last layout (H, W, C). Handles channel-first (C, H, W)."""
    if array.ndim == 3 and array.shape[0] < array.shape[1] and array.shape[0] < array.shape[2]:
        array = np.transpose(array, (1, 2, 0))
    return array


def _ensure_3channel(array: np.ndarray) -> np.ndarray:
    """Convert (H, W) or (H, W, C) to (H, W, 3). Must be channel-last already."""
    if array.ndim == 2:
        return np.stack([array] * 3, axis=-1)
    nc = array.shape[2]
    if nc == 1:
        return np.concatenate([array] * 3, axis=2)
    elif nc == 2:
        g = array.mean(axis=2).astype(array.dtype)
        return np.stack([g] * 3, axis=-1)
    elif nc == 3:
        return array
    else:
        g = array.mean(axis=2).astype(array.dtype)
        return np.stack([g] * 3, axis=-1)


def _to_rgb_uint8(array: np.ndarray) -> np.ndarray:
    """micro-SAM style: min-max normalisation → uint8 RGB."""
    array = _to_channel_last(array.astype(np.float32))
    lo, hi = array.min(), array.max()
    if hi > lo:
        array = (array - lo) / (hi - lo) * 255.0
    else:
        array = np.zeros_like(array)
    return _ensure_3channel(array.astype(np.uint8))


def _to_rgb_cellpose(array: np.ndarray) -> np.ndarray:
    """Cellpose style: per-channel percentile (1–99%) normalisation → uint8 RGB."""
    array = _ensure_3channel(_to_channel_last(array.astype(np.float32)))
    result = np.zeros_like(array)
    for c in range(3):
        ch = array[:, :, c]
        lo = np.percentile(ch, 1)
        hi = np.percentile(ch, 99)
        if hi > lo:
            result[:, :, c] = np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255)
    return result.astype(np.uint8)


def _to_rgb_cellsam(array: np.ndarray) -> np.ndarray:
    """CellSAM style: per-channel 99.9th-percentile clip + CLAHE → uint8 RGB."""
    array = _ensure_3channel(_to_channel_last(array.astype(np.float32)))
    # Step 1: 99.9th percentile clipping per channel
    for c in range(3):
        ch = array[:, :, c]
        non_zero = ch[np.nonzero(ch)]
        if len(non_zero) > 0:
            img_max = np.percentile(non_zero, 99.9)
            array[:, :, c] = np.clip(ch, 0, img_max)
    # Step 2: CLAHE per channel
    result = np.zeros_like(array)
    for c in range(3):
        X = array[:, :, c]
        if (X == X.flat[0]).all():
            result[:, :, c] = 0
            continue
        X = rescale_intensity(X, out_range=(0.0, 1.0))
        X = equalize_adapthist(X)
        result[:, :, c] = (X * 255).astype(np.uint8)
    return result.astype(np.uint8)


def _apply_norm(array: np.ndarray, norm_type: str) -> np.ndarray:
    """Apply named normalization to a raw array. Returns uint8 RGB (H, W, 3)."""
    if norm_type == 'cellpose':
        return _to_rgb_cellpose(array)
    elif norm_type == 'microsam':
        return _to_rgb_uint8(array)
    elif norm_type == 'cellsam':
        return _to_rgb_cellsam(array)
    elif norm_type == 'mouse_brain':
        return _preprocess_mouse_brain(array)
    else:
        raise ValueError(f"Unknown norm_type '{norm_type}'. Choose: cellpose, microsam, cellsam, mouse_brain")


def _preprocess_mouse_brain(array: np.ndarray) -> np.ndarray:
    """
    Mouse Brain specific preprocessing:
    - 4× bilinear upscaling (86×94 → 344×376)
    - Percentile normalization (1st-99th percentile → 0-255)
    """
    from PIL import Image as PILImage

    # Convert to channel-last if needed
    array = _to_channel_last(array.astype(np.float32))

    # Handle grayscale (H, W) or multi-channel
    if array.ndim == 2:
        h, w = array.shape
        pil_img = PILImage.fromarray(array)
    else:
        h, w = array.shape[:2]
        # Use first channel or average if multi-channel
        if array.shape[2] == 1:
            pil_img = PILImage.fromarray(array[:, :, 0])
        else:
            gray = array.mean(axis=2).astype(np.float32)
            pil_img = PILImage.fromarray(gray)

    # 4× bilinear upscaling
    pil_upscaled = pil_img.resize((w * 4, h * 4), PILImage.BILINEAR)
    img_upscaled = np.array(pil_upscaled, dtype=np.float32)

    # Percentile normalization (1st-99th → 0-255)
    p1, p99 = np.percentile(img_upscaled, [1, 99])
    if p99 > p1:
        img_normalized = np.clip((img_upscaled - p1) / (p99 - p1) * 255, 0, 255)
    else:
        img_normalized = np.zeros_like(img_upscaled)

    # Convert to uint8 RGB
    img_uint8 = img_normalized.astype(np.uint8)
    return np.stack([img_uint8] * 3, axis=-1)  # Make RGB


# Dataset-specific preprocessing mapping
DATASET_PREPROCESSING = {
    "LiveCELL": "cellpose",
    "TissueNet": "cellpose",
    "PlantSeg": "microsam",
    "Lizard": "cellpose",
    "2018_Data_Science": "cellpose",
    "Mouse_Brain": "mouse_brain",
    "Damond": "cellsam",
}


def load_and_preprocess_images(image_dir, image_names, dataset_name=None):
    imgs = []
    for name in image_names:
        path = os.path.join(image_dir, name)
        try:
            ext = os.path.splitext(name)[1].lower()
            if ext in ('.png', '.jpg', '.jpeg'):
                # Use PIL directly for non-TIFF formats
                img = Image.open(path).convert('RGB')
            else:
                array = tiff.imread(path)
                # Apply dataset-specific preprocessing
                if dataset_name == "Mouse_Brain":
                    img = Image.fromarray(_preprocess_mouse_brain(array))
                elif dataset_name in DATASET_PREPROCESSING:
                    norm_type = DATASET_PREPROCESSING[dataset_name]
                    img = Image.fromarray(_apply_norm(array, norm_type))
                else:
                    # Default to microsam (min-max) for unknown datasets
                    img = Image.fromarray(_to_rgb_uint8(array))
            imgs.append(img)
        except Exception as e:
            print(f"Could not read {path}: {e}")
            continue

    imgs_torch = [prep(img) for img in imgs]
    if torch.cuda.is_available():
        imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
    else:
        imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
    return imgs_torch

def load_and_preprocess_png(image_dir, image_names):
    imgs = []
    for name in image_names:
        path = os.path.join(image_dir, name)
        try:
            img = Image.open(path).convert('RGB')  # Ensure 3-channel RGB
            imgs.append(img)
        except Exception as e:
            print(f"Could not read {path}: {e}")
            continue

    imgs_torch = [prep(img) for img in imgs]
    if torch.cuda.is_available():
        imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
    else:
        imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]

    return imgs_torch