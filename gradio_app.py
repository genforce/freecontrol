import os.path
import time
from typing import Dict, List

import gradio as gr
import numpy as np
import torch
import yaml
from PIL import Image
from omegaconf import OmegaConf

from libs.utils.utils import merge_sweep_config
from libs.model import make_pipeline
from libs.model.module.scheduler import CustomDDIMScheduler
from libs.utils.controlnet_processor import make_processor


def freecontrol_generate(condition_image, prompt, scale, ddim_steps, sd_version,
                         model_ckpt, pca_guidance_steps, pca_guidance_components,
                         pca_guidance_weight, pca_guidance_normalized,
                         pca_masked_tr, pca_guidance_penalty_factor, pca_warm_up_step, pca_texture_reg_tr,
                         pca_texture_reg_factor,
                         negative_prompt, seed, paired_objs,
                         pca_basis_dropdown, inversion_prompt, condition, img_size, **kwargs):
    control_type = condition

    if not control_type == "None":
        processor = make_processor(control_type.lower())
    else:
        processor = lambda x: Image.open(x).convert("RGB") if type(x) == str else x

    # get the config
    model_path = model_dict[sd_version][model_ckpt]['path']
    # define kwargs
    gradio_update_parameter = {
        # Stable Diffusion Generation Configuration ,
        'sd_config--guidance_scale': scale,
        'sd_config--steps': ddim_steps,
        'sd_config--seed': seed,
        'sd_config--dreambooth': False,
        'sd_config--prompt': prompt,
        'sd_config--negative_prompt': negative_prompt,
        'sd_config--obj_pairs': str(paired_objs),
        'sd_config--pca_paths': [pca_basis_dict[sd_version][model_ckpt][pca_basis_dropdown]],

        'data--inversion--prompt': inversion_prompt,
        'data--inversion--fixed_size': [img_size, img_size],

        # PCA Guidance Parameters
        'guidance--pca_guidance--end_step': int(pca_guidance_steps * ddim_steps),
        'guidance--pca_guidance--weight': pca_guidance_weight,
        'guidance--pca_guidance--structure_guidance--n_components': pca_guidance_components,
        'guidance--pca_guidance--structure_guidance--normalize': bool(pca_guidance_normalized),
        'guidance--pca_guidance--structure_guidance--mask_tr': pca_masked_tr,
        'guidance--pca_guidance--structure_guidance--penalty_factor': pca_guidance_penalty_factor,

        'guidance--pca_guidance--warm_up--apply': True if pca_warm_up_step > 0 else False,
        'guidance--pca_guidance--warm_up--end_step': int(pca_warm_up_step * ddim_steps),
        'guidance--pca_guidance--appearance_guidance--apply': True if pca_texture_reg_tr > 0 else False,
        'guidance--pca_guidance--appearance_guidance--tr': pca_texture_reg_tr,
        'guidance--pca_guidance--appearance_guidance--reg_factor': pca_texture_reg_factor,

        # Cross Attention Guidance Parameters
        'guidance--cross_attn--end_step': int(pca_guidance_steps * ddim_steps),
        'guidance--cross_attn--weight': 0,

    }

    input_config = gradio_update_parameter

    # Load base config
    base_config = yaml.load(open("config/base.yaml", "r"), Loader=yaml.FullLoader)
    # Update the Default config by gradio config
    config = merge_sweep_config(base_config=base_config, update=input_config)
    config = OmegaConf.create(config)

    # set the correct pipeline
    pipeline_name = "SDPipeline"

    pipeline = make_pipeline(pipeline_name,
                             model_path,
                             torch_dtype=torch.float16).to('cuda')
    pipeline.scheduler = CustomDDIMScheduler.from_pretrained(model_path, subfolder="scheduler")

    # create a inversion config
    inversion_config = config.data.inversion

    # Processor the condition image
    img = processor(condition_image)
    # flip the color for the scribble and canny: black background to white background
    if control_type == "scribble" or control_type == "canny":
        img = Image.fromarray(255 - np.array(img))

    condition_image_latents = pipeline.invert(img=img, inversion_config=inversion_config)

    inverted_data = {"condition_input": [condition_image_latents], }

    g = torch.Generator()
    g.manual_seed(config.sd_config.seed)

    img_list = pipeline(prompt=config.sd_config.prompt,
                        negative_prompt=config.sd_config.negative_prompt,
                        num_inference_steps=config.sd_config.steps,
                        generator=g,
                        config=config,
                        inverted_data=inverted_data)[0]

    # Display the result：
    # if the control type is not none, then we display [condition_image, output_image, output_image_with_control]
    # if the control type is none, then we display [condition_image, output_image]
    if control_type != "None":
        img_list.insert(0, img)
    return img_list


def change_sd_version(sd_version):
    model_ckpt_list: List = list(model_dict[sd_version].keys())
    model_ckpt = gr.Radio(model_ckpt_list, label="Select a Model", value=model_ckpt_list[0])
    model_name = model_ckpt_list[0]

    pca_basis = change_model_ckpt(sd_version, model_name)
    return model_ckpt, pca_basis


def change_model_ckpt(sd_version, model_name):
    pca_basis_list: List = list(pca_basis_dict[sd_version][model_name].keys()) if pca_basis_dict[sd_version][
                                                                                      model_name].keys() is not None else []

    if len(pca_basis_list) != 0:
        pca_basis = gr.Dropdown(label="Select a PCA Basis",
                                choices=pca_basis_list, value=pca_basis_list[0])
    else:
        pca_basis = gr.Dropdown(label="Select a PCA Basis",
                                choices=pca_basis_list)

    return pca_basis


def load_ckpt_pca_list(config_path='config/gradio_info.yaml'):
    """
    Load the checkpoint and pca basis list from the config file
    :param config_path:
    :return:
    models : Dict: The dictionary of the model checkpoints
    pca_basis_dict : List : The list of the pca basis

    """

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist")

    # load from config
    with open(config_path, 'r') as f:
        gradio_config = yaml.safe_load(f)

    models: Dict = gradio_config['checkpoints']
    pca_basis_dict: Dict = dict()

    # remove non-exist model
    for model_version in list(models.keys()):
        for model_name in list(models[model_version].keys()):
            if "naive" not in model_name and not os.path.isfile(models[model_version][model_name]["path"]):
                models[model_version].pop(model_name)
            else:
                # Add the path of PCA basis to the pca_basis dict
                basis_dict = models[model_version][model_name]["pca_basis"]
                for key in list(basis_dict.keys()):
                    if not os.path.isfile(basis_dict[key]):
                        basis_dict.pop(key)
                if model_version not in pca_basis_dict.keys():
                    pca_basis_dict[model_version]: Dict = dict()
                if model_name not in pca_basis_dict[model_version].keys():
                    pca_basis_dict[model_version][model_name]: Dict = dict()
                pca_basis_dict[model_version][model_name].update(basis_dict)

    return models, pca_basis_dict


def main():
    global model_dict, pca_basis_dict
    # Load checkpoint and pca basis list
    model_dict, pca_basis_dict = load_ckpt_pca_list()

    block = gr.Blocks()
    with block as demo:
        with gr.Row():
            gr.Markdown(
                "## FreeControl: Training-Free Spatial Control of Any Text-to-Image Diffusion Model with Any Condition\n"
                "#### Following the steps to generate the images: \n"
                "#### 1. Select a SD Version, Model Checkpoint, and PCA Basis\t\t\t"
                " 2. Input the condition image, prompt, inversion prompt, and subject pairs\n"
                "#### 3. Select the control type and set the guidance parameters\t\t\t"
                " 4. Click the Run button to generate the images\n")
        with gr.Row():
            with gr.Column():
                # Add condition image from user input
                input_image = gr.Image(label="Input Condition Image", type="pil", interactive=True,
                                       value=Image.open("dataset/example_dog.jpg") if os.path.exists("dataset/example_dog.jpg") else None)

                # Select the SD Version, Model Checkpoint and PCA Basis
                sd_version = gr.Radio(list(model_dict.keys()), label="Select a Base Model", value="1.5")
                model_ckpt = gr.Radio(list(model_dict[sd_version.value].keys()), label="Select a Model",
                                      value=list(model_dict[sd_version.value].keys())[0])

                pca_basis_list: List = list(pca_basis_dict[sd_version.value][model_ckpt.value].keys()) if \
                    pca_basis_dict[sd_version.value][model_ckpt.value].keys() is not None else []

                pca_basis = gr.Dropdown(label="Select Semantic Bases",
                                        choices=pca_basis_list, )
                print(pca_basis.value)
                sd_version.change(fn=change_sd_version, inputs=sd_version,
                                  outputs=[model_ckpt, pca_basis],
                                  scroll_to_output=True)
                model_ckpt.change(fn=change_model_ckpt, inputs=[sd_version, model_ckpt],
                                  outputs=pca_basis,
                                  scroll_to_output=True)

            with gr.Column():
                prompt = gr.Textbox(label="Generation Prompt: prompt to generate target image",
                                    value="A photo of a lion, in the desert, best quality, extremely detailed")
                inversion_prompt = gr.Textbox(label="Inversion Prompt to invert the condition image",
                                              value="A photo of a dog")
                paired_objs = gr.Textbox(
                    label="Paired subject: Please selected the paired subject from the inverson prompt and generation prompt."
                          "Then input in the format like (obj from inversion prompt; obj from generation prompt)"
                          "e.g. (dog; lion)",
                    value="(dog; lion)")
                run_button = gr.Button(value="Run")
                with gr.Accordion("options", open=True):
                    scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
                    ddim_steps = gr.Slider(label="DDIM Steps", minimum=1, maximum=200, value=200, step=1)
                    img_size = gr.Slider(label="Image Size", minimum=256, maximum=1024, value=512, step=64)

                    condition = gr.Radio(
                        choices=["None", "Scribble", "Depth", "Hed", "Seg", "Canny", "Normal", "Openpose"],
                        label="Condition Type: extract condition on the input image", value="None")

                    seed = gr.Slider(label="Seed", minimum=0, maximum=100000, value=2028, step=1)

                    # PCA Q,K guidance parameters
                    pca_guidance_steps = gr.Slider(label="PCA Guidance End Steps", minimum=0, maximum=1, value=0.6,
                                                   step=0.1)
                    pca_guidance_components = gr.Slider(label="Structure Guidance: Number of Component", minimum=-1,
                                                        maximum=64,
                                                        value=64, step=1)
                    pca_guidance_weight = gr.Slider(label="Structure Guidance: Weight", minimum=0, maximum=1000, value=600,
                                                    step=50)

                with gr.Accordion("Advanced Options (dont need to change)", open=False):
                    # Negative Prompt
                    negative_prompt = gr.Textbox(label="Negative Prompt: negative prompt with classifier free guidance",
                                                 value="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality")
                    # Advanced PCA guidance options
                    pca_guidance_normalized = gr.Checkbox(label="PCA Guidance Normalized", value=True,
                                                          info="Enable normalization")
                    pca_masked_tr = gr.Slider(label="Cross-attention Mask Threshold", minimum=0, maximum=1, value=0.3, step=0.1)
                    pca_guidance_penalty_factor = gr.Slider(label="Structure Guidance: Background Penalty Factor", minimum=0, maximum=100,
                                                            value=10, step=0.00001)
                    pca_warm_up_step = gr.Slider(label="Guidance Warm Up Step", minimum=0, maximum=1, value=0.05, step=0.05)
                    pca_texture_reg_tr = gr.Slider(label="PCA Appearance Guidance Threshold", minimum=0, maximum=1,
                                                   value=0.5, step=0.1)
                    pca_texture_reg_factor = gr.Slider(label="PCA Appearance Guidance Factor", minimum=0, maximum=1,
                                                       value=0.1, step=0.1)

            with gr.Column():
                gr.Markdown("#### Output Images: \n"
                            "If the control type is not none, then we display [condition image, output image, output image without control]\n"
                            "If the control type is none, then we display [output image, output image without control]")
                result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", columns=2,
                                            height='auto')  # .style(columns=2, height='auto')

        ips = [input_image, prompt, scale, ddim_steps, sd_version,
               model_ckpt, pca_guidance_steps, pca_guidance_components, pca_guidance_weight,
               pca_guidance_normalized,
               pca_masked_tr, pca_guidance_penalty_factor, pca_warm_up_step, pca_texture_reg_tr, pca_texture_reg_factor,
               negative_prompt, seed, paired_objs,
               pca_basis, inversion_prompt, condition, img_size]

        run_button.click(fn=freecontrol_generate, inputs=ips, outputs=[result_gallery])

    block.launch(server_name='0.0.0.0', share=False, server_port=9989)


if __name__ == '__main__':
    main()
