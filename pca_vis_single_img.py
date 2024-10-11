import time

import torch
import yaml
from PIL import Image
from libs.model import make_pipeline
from libs.model.module.scheduler import CustomDDIMScheduler
from omegaconf import OmegaConf
import os
from baseline_scripts.controlNet.processor import make_processor
import torchvision.transforms.functional as F
import numpy as np
def concat_images_and_tensors(images, tensors):
    # 验证输入
    if not (isinstance(images, list) and all(isinstance(img, Image.Image) for img in images)):
        raise TypeError("images must be a list of PIL.Image.Image objects")
    if not (isinstance(tensors, torch.Tensor) and tensors.dim() == 4):
        raise TypeError("tensors must be a 4-dimensional torch.Tensor")
    if len(images) != tensors.size(0):
        raise ValueError("The length of images and the first dimension of tensors must be the same")

    # 调整tensor的大小和范围
    tensors = (tensors - tensors.min()) / (tensors.max() - tensors.min())  # 归一化
    tensors = F.resize(tensors, (512, 512), interpolation=Image.NEAREST)  # resize

    # 调整图片的大小
    images = [img.resize((512, 512), resample=Image.BILINEAR) for img in images]

    # 拼接所有的图片
    image_row = Image.new('RGB', (512 * len(images), 512))
    for i, img in enumerate(images):
        image_row.paste(img, (512 * i, 0))

    # 拼接所有的tensor的可视化
    tensor_row = Image.new('RGB', (512 * len(images), 512))
    for i, tensor in enumerate(tensors):
        tensor_row.paste(F.to_pil_image(tensor), (512 * i, 0))

    # 拼接图片和tensor的可视化
    final_image = Image.new('RGB', (512 * len(images), 512 * 2))
    final_image.paste(image_row, (0, 0))
    final_image.paste(tensor_row, (0, 512))
    return final_image

if __name__ == '__main__':

    config = yaml.load(open("config/baseline/pca_guidance.yaml", "r"), Loader=yaml.FullLoader)
    config["sd_config"]["pca_paths"] = [
        # 'dataset/basis/2.1_base_v1/2.1_base_naive/bedroom/step_200_sample_20/pca_info.pt'
        #'dataset/basis/2.1_base/2.1_base_naive/person/step_200_sample_20/pca_info.pt'
        'dataset/basis/2.1_base/2.1_base_naive/cat1/step_200_sample_20/pca_info.pt'
    ]
    config["data"]["inversion"]= {
        "target_folder": 'dataset/latent',
        "num_inference_steps": 999,
        "method": 'DDIM',                # choice from ['DDIM'|'NTI'|'NPT']
        "fixed_size": [512,512],              # Set to null to disable fixed size, otherwise set to the fixed size (h,w) of the target image
        "prompt": "A cat",
        "select_objects": "cat",
        "policy": "share",               # choice from ['share'|'separate']
        "sd_model": '2.1_base_naive',
    }

    config = OmegaConf.create(config)

    sd_config  = config.sd_config


    pipeline = make_pipeline("SDPipeline",
                             "stabilityai/stable-diffusion-2-1-base",
                             # 'models/2.1/Cute_RichStyle',
                             safetensors=False,
                             safety_checker=None,
                             torch_dtype = torch.float16
                             ).to('cuda')
    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.scheduler = CustomDDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler")

    # create a inversion config
    inversion_config  = config.data.inversion

    # load the image
    control_type = "normal"
    if not control_type == "None":
        processor = make_processor(control_type)
    else:
        processor = lambda x: Image.open(x).convert("RGB")

    #
    img_path = "dataset/others/iron_man.jpg"
    img_path = "dataset/controlnet/human_line.png"
    #img_path = "dataset/others/3d_cartoon_gril.jpg"
    #img_path = "dataset/controlnet/person_car.jpg"
    img_path = "dataset/others/bedroom4.jpg"
    img_path = "dataset/others/man1.jpg"
    img_path = "dataset/others/cat_photo_2.jpg"
    #img_path = "dataset/sicheng_collect/pose/room_/img/img_001.png"
    img = processor(img_path)

    if control_type == "scribble" or control_type == "canny":
        img = Image.fromarray(255 - np.array(img))

    start_time = time.time()
    data_samples_pose = pipeline.invert(img = img, inversion_config = inversion_config)
    end_time = time.time()
    # print the time in seconds, with only 2 decimal places
    print("Time elapsed: {:.2f} seconds".format(end_time - start_time))

    data_samples = {
        "examplar": [data_samples_pose],
        "appearance": None,
    }
    g = torch.Generator()
    g.manual_seed(2094)
    pca_dict = pipeline.compute_score(prompt = "A photo of man, in a street",
                          num_inference_steps = 50,
                          generator = g,
                          config = config,
                          data_samples = data_samples )
    image_list = [ data_samples_pose['pil_img'] ]
    root_dir = f"experiments/pca_vis/2.1_base_naive/cat_id1_{control_type}"
    for key,value in pca_dict.items():
        step = key
        for feat_name in value.keys():
            for block_name in value[feat_name].keys():
                folder_name = os.path.join(root_dir, feat_name, block_name)
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name,exist_ok=True)
                score = value[feat_name][block_name]['score']
                # scores = score.chunk(2, dim=1)
                # for score_id, score in enumerate(scores):
                #     # print(score.shape)
                #     # exit()
                #     final_img = concat_images_and_tensors(image_list, score)
                #     final_img.save(os.path.join(folder_name, "step_{:02d}_id{:02d}.png".format(int(step),score_id)))
                # # # print(score.shape)
                final_img = concat_images_and_tensors(image_list, score)
                final_img.save(os.path.join(folder_name, str(step) + ".png"))

    print("Done")
