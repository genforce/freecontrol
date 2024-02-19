import argparse
import os

import torch
import yaml
from omegaconf import OmegaConf

from libs.model import make_pipeline
from libs.model.module.scheduler import CustomDDIMScheduler


def main(args):
    gradio_info = yaml.load(open('config/gradio_info.yaml', "r"), Loader=yaml.FullLoader)
    models_info = gradio_info["checkpoints"]
    if args.sd_version not in models_info.keys():
        raise ValueError(f"Model {args.sd_version} not found in the model list: {list(models_info.keys())}.")
    model_ckpt_list = models_info[args.sd_version]

    if args.model_name not in model_ckpt_list.keys():
        raise ValueError(
            f"Stable Diffusion version {args.model_name} not found in the model {args.sd_version} list: {list(model_ckpt_list.keys())}.")
    model_path = model_ckpt_list[args.model_name]['path']

    config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    config = OmegaConf.create(config)
    pipeline_name = "SDPipeline"
    pipeline = make_pipeline(pipeline_name,
                             model_path,
                             torch_dtype=torch.float16
                             ).to('cuda')
    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.scheduler = CustomDDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
    g = torch.Generator()
    g.manual_seed(args.seed)
    pipeline.sample_semantic_bases(prompt=args.prompt,
                                   negative_prompt=args.negative_prompt,
                                   generator=g,
                                   num_inference_steps=args.num_steps,
                                   height=args.height,
                                   width=args.width,
                                   num_images_per_prompt=args.num_images,
                                   num_batch=args.num_batch,
                                   config=config,
                                   num_save_basis=args.num_bases,
                                   num_save_steps=args.num_save_steps,
                                   )
    sd_version = args.sd_version
    model_name = args.model_name
    output_class = args.output_class

    id = 0
    output_path = f"dataset/basis/{sd_version}/{model_name}/{output_class}/step_{args.num_steps}_sample_{int(args.num_images * args.num_batch)}_id_{id}"
    while os.path.exists(output_path):
        id += 1
        output_path = f"dataset/basis/{sd_version}/{model_name}/{output_class}/step_{args.num_steps}_sample_{int(args.num_images * args.num_batch)}_id_{id}"
    os.makedirs(output_path, exist_ok=True)
    pca_info = pipeline.pca_info
    torch.save(pca_info, f"{output_path}/pca_info.pt")

    if args.log:
        pca_basis_name = f"{output_class}_step_{args.num_steps}_sample_{int(args.num_images * args.num_batch)}_id_{id}"
        if 'pca_basis' not in gradio_info['checkpoints'][sd_version][model_name].keys():
            gradio_info['checkpoints'][sd_version][model_name] = {}
        gradio_info['checkpoints'][sd_version][model_name]['pca_basis'].update({pca_basis_name: f"{output_path}/pca_info.pt"})
        with open('config/gradio_info.yaml', 'w') as f:
            yaml.dump(gradio_info, f)
        print("Updated gradio_info.yaml")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate images using the provided configuration.')
    parser.add_argument('--config_path', type=str, default="config/base.yaml",
                        help='Path to the configuration YAML file.')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed.')
    parser.add_argument('--prompt', type=str, default="A photo of a cat with simple background, best quality, "
                                                      "extremely detailed", help='Image generation prompt.')
    parser.add_argument('--negative_prompt', type=str, default="", help='Negative image generation prompt.')
    parser.add_argument('--num_steps', type=int, default=199, help='Number of inference steps.')
    parser.add_argument('--height', type=int, default=512, help='Image height.')
    parser.add_argument('--width', type=int, default=512, help='Image width.')
    parser.add_argument('--num_images', type=int, default=5, help='Number of images per prompt.')
    parser.add_argument('--num_batch', type=int, default=2, help='Batch size.')
    parser.add_argument('--output_class', type=str, default="toy_bear", help='Output class.')
    parser.add_argument('--sd_version', type=str, default=1.5, help='Stable Diffusion version.')
    parser.add_argument('--model_name', type=str, default="", help='Model name.')
    parser.add_argument('--num_bases', type=int, default=64, help='Number of PCA bases to save.')
    parser.add_argument('--num_save_steps', type=int, default=120, help='Number of steps to save the PCA bases.')
    parser.add_argument('--log', action='store_true', help='Log to gradio_info.yaml file')

    args = parser.parse_args()

    main(args)
