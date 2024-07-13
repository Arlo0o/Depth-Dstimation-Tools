
from __future__ import print_function, division
import argparse
import os
from pyrsistent import v
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
print(torch.cuda.is_available())
import random
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import sys
import math
import cv2
from PIL import Image
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.utils.data
from datasets import __datasets__
import gc
import skimage
import skimage.io
import skimage.transform
from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
import util.io
from torchvision.transforms import Compose


def trans_color(gray):
    gray = gray.reshape(gray.shape[0], gray.shape[1], 1).astype('uint8')
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    out = cv2.applyColorMap(out, cv2.COLORMAP_JET)
    return out



if __name__ == '__main__':

    ##########################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_path", default="./images/in_r/", help="folder with input images"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="./images/out_r",
        help="folder for output images",
    )
    parser.add_argument(
        "-m", "--model_weights", default=None, help="path to model weights"
    )
    parser.add_argument(
        "-t",
        "--model_type",
        default="dpt_hybrid",
        help="model type [dpt_large|dpt_hybrid|midas_v21]",
    )
    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")
    parser.set_defaults(optimize=True)
    args = parser.parse_args()
    default_models = {
        "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
        "dpt_large": "dpt/weights/dpt_large-midas-2f21e586.pt",
    }
    
    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load network
    if args.model_type == "dpt_large":  # DPT-Large
        net_w = net_h = 384
        model = DPTDepthModel(
            path="./dpt/weights/dpt_large-midas-2f21e586.pt",
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif args.model_type == "dpt_hybrid":
        net_w = net_h = 384
        model = DPTDepthModel(
            path= "./dpt/weights/dpt_hybrid-midas-501f0c75.pt",
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )
    model.eval()
    model = model.to(memory_format=torch.channels_last)
    model = model.half()
    model.to(device)
    img = util.io.read_image("./dpt/pic/1.png")
    img_input = transform({"image": img})["image"]
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        print(sample.size() )
        if args.optimize == True and device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()
    
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
        )
    util.io.write_depth("./dpt/pic/out", prediction, bits=2)
    
    
    # img = cv2.imread('./dpt/pic/out.png', 0)
    # img = cv2.convertScaleAbs(img, alpha=0.5, beta=-20)
    # output0 = trans_color(img)
    # cv2.imwrite('./dpt/pic/outc.png', output0)
    
    
    
    
    ###################-----------------------------------
    # imgL = cv2.imread('./images/videoTest/L/00.png', 0)
    # imgR = cv2.imread('./images/videoTest/R/00.png', 0)
    # # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
    # stereo = cv2.StereoSGBM_create(
    #     minDisparity=0,
    #     numDisparities=64,  # max_disp has to be dividable by 16 f. E. HH 192, 256
    #     blockSize=15,
    #     P1=8 * 3 * 3 ** 2,
    #     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    #     P2=32 * 3 * 3 ** 2,
    #     disp12MaxDiff=1,
    #     uniquenessRatio=5,
    #     speckleWindowSize=0,
    #     speckleRange=2,
    #     preFilterCap=63,
    #     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    # )
    # pred = stereo.compute(imgL, imgR).astype(np.float32)
    # pred = pred[0:480, 60:640]

    # cv2.imwrite('./images/out1/1.png', pred)
    # pred = cv2.convertScaleAbs(pred, alpha=0.2, beta=-5)
    # out0 = trans_color(pred)
    # cv2.imwrite('./images/out2/1.png', out0)
    
    