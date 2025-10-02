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


def load_and_preprocess_images(image_dir, image_names):
    imgs = []
    for name in image_names:
        path = os.path.join(image_dir, name)
        try:
            array = tiff.imread(path)
            if array.ndim == 2:  # H x W, convert to 3 channels
                array = np.stack([array]*3, axis=-1)
            elif array.shape[0] <= 3 and array.ndim == 3:  # C x H x W
                array = np.transpose(array, (1, 2, 0))
            img = Image.fromarray(array.astype(np.uint8))
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