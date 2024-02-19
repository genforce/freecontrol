from .utils import load_image
import torch
import argparse
import yaml
from PIL import Image
from controlnet_aux import OpenposeDetector
from controlnet_aux import HEDdetector
from controlnet_aux import MLSDdetector
import os
import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline
import copy
import cv2
from abc import ABC, abstractmethod

palette = np.asarray([
    [0, 0, 0],
    [120, 120, 120],
    [180, 120, 120],
    [6, 230, 230],
    [80, 50, 50],
    [4, 200, 3],
    [120, 120, 80],
    [140, 140, 140],
    [204, 5, 255],
    [230, 230, 230],
    [4, 250, 7],
    [224, 5, 255],
    [235, 255, 7],
    [150, 5, 61],
    [120, 120, 70],
    [8, 255, 51],
    [255, 6, 82],
    [143, 255, 140],
    [204, 255, 4],
    [255, 51, 7],
    [204, 70, 3],
    [0, 102, 200],
    [61, 230, 250],
    [255, 6, 51],
    [11, 102, 255],
    [255, 7, 71],
    [255, 9, 224],
    [9, 7, 230],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [7, 255, 224],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [255, 122, 8],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
    [31, 255, 0],
    [255, 31, 0],
    [255, 224, 0],
    [153, 255, 0],
    [0, 0, 255],
    [255, 71, 0],
    [0, 235, 255],
    [0, 173, 255],
    [31, 0, 255],
    [11, 200, 200],
    [255, 82, 0],
    [0, 255, 245],
    [0, 61, 255],
    [0, 255, 112],
    [0, 255, 133],
    [255, 0, 0],
    [255, 163, 0],
    [255, 102, 0],
    [194, 255, 0],
    [0, 143, 255],
    [51, 255, 0],
    [0, 82, 255],
    [0, 255, 41],
    [0, 255, 173],
    [10, 0, 255],
    [173, 255, 0],
    [0, 255, 153],
    [255, 92, 0],
    [255, 0, 255],
    [255, 0, 245],
    [255, 0, 102],
    [255, 173, 0],
    [255, 0, 20],
    [255, 184, 184],
    [0, 31, 255],
    [0, 255, 61],
    [0, 71, 255],
    [255, 0, 204],
    [0, 255, 194],
    [0, 255, 82],
    [0, 10, 255],
    [0, 112, 255],
    [51, 0, 255],
    [0, 194, 255],
    [0, 122, 255],
    [0, 255, 163],
    [255, 153, 0],
    [0, 255, 10],
    [255, 112, 0],
    [143, 255, 0],
    [82, 0, 255],
    [163, 255, 0],
    [255, 235, 0],
    [8, 184, 170],
    [133, 0, 255],
    [0, 255, 92],
    [184, 0, 255],
    [255, 0, 31],
    [0, 184, 255],
    [0, 214, 255],
    [255, 0, 112],
    [92, 255, 0],
    [0, 224, 255],
    [112, 224, 255],
    [70, 184, 160],
    [163, 0, 255],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [255, 0, 143],
    [0, 255, 235],
    [133, 255, 0],
    [255, 0, 235],
    [245, 0, 255],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 41, 255],
    [0, 255, 204],
    [41, 0, 255],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [122, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [0, 133, 255],
    [255, 214, 0],
    [25, 194, 194],
    [102, 255, 0],
    [92, 0, 255],
])


processor = {}
def register_processor(name):
    def decorator(fn):
        processor[name] = fn
        return fn
    return decorator

def make_processor(name):
    return processor[name]()

class Processor(ABC):
    @abstractmethod
    def __call__(self, img_path):
        pass
    @abstractmethod
    def get_controlnet_id(self):
        pass

@register_processor("openpose")
class Openposedetector(Processor):

    def get_controlnet_id(self):
        return "lllyasviel/sd-controlnet-openpose"

    def __call__(self, img_path):
        openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        image = load_image(img_path)
        image = openpose(image)
        condition_image = image
        return condition_image


@register_processor("depth")
class DepthMapDetector(Processor):

    def get_controlnet_id(self):
        return "lllyasviel/sd-controlnet-depth"

    def __call__(self, img_path):
        depth_estimator = pipeline('depth-estimation')
        image = load_image(img_path)
        image = depth_estimator(image)['depth']
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)

        condition_image = copy.deepcopy(image)
        return condition_image

@register_processor("hed")
class HEDDetector(Processor):


    def get_controlnet_id(self):
        return "lllyasviel/sd-controlnet-hed"

    def __call__(self, img_path):
        hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")
        image = load_image(img_path)
        image = hed(image)
        condition_image = image
        return condition_image

@register_processor("mlsd")
class MLSDDetector(Processor):

    def get_controlnet_id(self):
        return "lllyasviel/sd-controlnet-mlsd"

    def __call__(self, img_path):
        mlsd = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
        image = load_image(img_path)
        image = mlsd(image)
        condition_image = image
        return condition_image

@register_processor("canny")
class CannyEdgeDetector(Processor):

    def get_controlnet_id(self):
        return "lllyasviel/sd-controlnet-canny"

    def __call__(self,img_path):
        image = load_image(img_path)
        image = np.array(image)

        low_threshold = 100
        high_threshold = 200

        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)

        condition_image = copy.deepcopy(image)
        return condition_image

@register_processor("normal")
class NormalMapDetector(Processor):

    def get_controlnet_id(self):
        return "fusing/stable-diffusion-v1-5-controlnet-normal"

    def __call__(self, img_path):
        image = load_image(img_path)
        depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")

        image = depth_estimator(image)['predicted_depth'][0]

        image = image.numpy()

        image_depth = image.copy()
        image_depth -= np.min(image_depth)
        image_depth /= np.max(image_depth)

        bg_threhold = 0.4

        x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        x[image_depth < bg_threhold] = 0

        y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        y[image_depth < bg_threhold] = 0

        z = np.ones_like(x) * np.pi * 2.0

        image = np.stack([x, y, z], axis=2)
        image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
        image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        condition_image = copy.deepcopy(image)
        return condition_image

@register_processor("scribble")
class ScribbleDetector(Processor):

    def get_controlnet_id(self):
        return  "lllyasviel/sd-controlnet-scribble"

    def __call__(self, img_path):
        hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
        image = load_image(img_path)
        image = hed(image, scribble=True)
        fliped_image = Image.fromarray(255 - np.array(image))
        condition_image = copy.deepcopy(image)
        return condition_image

@register_processor("seg")
class SegMapDetector(Processor):

    def get_controlnet_id(self):
        return "lllyasviel/sd-controlnet-seg"

    def __call__(self, img_path):
        from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
        image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
        image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

        image = load_image(img_path)
        pixel_values = image_processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = image_segmentor(pixel_values)

        seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3

        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color

        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color

        color_seg = color_seg.astype(np.uint8)

        image = Image.fromarray(color_seg)
        condition_image = copy.deepcopy(image)
        return condition_image

