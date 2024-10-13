import argparse
import os
import time
import yaml

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from libs.model import make_pipeline
from libs.model.module.scheduler import CustomDDIMScheduler
from viz.processor import make_processor
import torchvision.transforms.functional as F


def concat_images_and_tensors(images, tensors):
    # validate inputs
    if not (isinstance(images, list) and all(isinstance(img, Image.Image) for img in images)):
        raise TypeError("images must be a list of PIL.Image.Image objects")
    if not (isinstance(tensors, torch.Tensor) and tensors.dim() == 4):
        raise TypeError("tensors must be a 4-dimensional torch.Tensor")
    if len(images) != tensors.size(0):
        raise ValueError("The length of images and the first dimension of tensors must be the same")

    # normalize and resize
    tensors = (tensors - tensors.min()) / (tensors.max() - tensors.min())
    tensors = F.resize(tensors, (512, 512), interpolation=Image.NEAREST)
    images = [img.resize((512, 512), resample=Image.BILINEAR) for img in images]

    # combine all images and visualizations
    image_row = Image.new("RGB", (512 * len(images), 512))
    for i, img in enumerate(images):
        image_row.paste(img, (512 * i, 0))

    tensor_row = Image.new("RGB", (512 * len(images), 512))
    for i, tensor in enumerate(tensors):
        tensor_row.paste(F.to_pil_image(tensor), (512 * i, 0))

    final_image = Image.new("RGB", (512 * len(images), 512 * 2))
    final_image.paste(image_row, (0, 0))
    final_image.paste(tensor_row, (0, 512))
    
    return final_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="1.5", help="Diffusion model name")
    parser.add_argument("--pca-path", type=str, help="path to semantic bases from PCA")
    parser.add_argument("--img-path", type=str, help="Image path")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--img-type", type=str, default="rgb", help="Image type")
    parser.add_argument("--inv-prompt", type=str, help="Text prompt for inversion")
    parser.add_argument("--gen-prompt", type=str, help="Text prompt for generation")
    parser.add_argument("--object", type=str, help="Object type")
    args = parser.parse_args()

    # currently only support 1.5 and 2.1 base
    if args.model == "1.5":
        model_name = "sd-legacy/stable-diffusion-v1-5"
    elif args.model == "2.1_base":
        model_name = "stabilityai/stable-diffusion-2-1-base"
    else:
        raise ValueError(f"Model {args.model} currently not supported.")

    # load configs
    config = yaml.load(open("config/base.yaml", "r"), Loader=yaml.FullLoader)
    assert os.path.exists(args.pca_path)
    config["sd_config"]["pca_paths"] = [args.pca_path]
    config["data"]["inversion"] = {
        "target_folder": "dataset/latent",
        "num_inference_steps": 999,
        "method": "DDIM",
        "fixed_size": [512, 512],
        "prompt": args.inv_prompt,
        "select_objects": args.object,
        "policy": "share",
        "sd_model": f"{args.model}_naive",
    }
    config = OmegaConf.create(config)

    # load pipeline
    pipeline = make_pipeline(
        "SDPipeline",
        model_name,
        safetensors=False,
        safety_checker=None,
        torch_dtype=torch.float16
    ).to("cuda")
    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.scheduler = CustomDDIMScheduler.from_pretrained(
        model_name,
        subfolder="scheduler",
    )

    # load image
    if not args.img_type != "rgb":
        processor = make_processor(args.img_type)
    else:
        processor = lambda x: Image.open(x).convert("RGB")
    img_name = ".".join(os.path.basename(args.img_path).split(".")[:-1])
    img = processor(args.img_path)
    if args.img_type in ("scribble", "canny"):
        img = Image.fromarray(255 - np.array(img))

    # run inversion to generate features
    start_time = time.time()
    data_samples_pose = pipeline.invert(img=img, inversion_config=config.data.inversion)
    print(f"Time elapsed: {(time.time() - start_time):.2f} seconds")

    # project onto semantic bases from PCA
    data_samples = {
        "examplar": [data_samples_pose],
        "appearance": None,
    }
    g = torch.Generator()
    g.manual_seed(2094)
    pca_dict = pipeline.compute_score(
        prompt=args.gen_prompt,
        num_inference_steps=50,
        generator=g,
        config=config,
        data_samples=data_samples,
    )

    # save visualization
    image_list = [data_samples_pose["pil_img"]]
    root_dir = os.path.join(args.output_dir, f"{img_name}_{args.img_type}")
    for key, value in pca_dict.items():
        step = key
        for feat_name in value.keys():
            for block_name in value[feat_name].keys():
                folder_name = os.path.join(root_dir, feat_name, block_name)
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name,exist_ok=True)
                score = value[feat_name][block_name]["score"]
                final_img = concat_images_and_tensors(image_list, score)
                final_img.save(os.path.join(folder_name, f"{str(step)}.png"))

    print("Success")
