import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from diffusers import StableDiffusionXLPipeline, DDIMInverseScheduler
from diffusers.utils import BaseOutput
from numpy import deprecate

from libs.utils.utils import *
from .module import prep_unet_conv, prep_unet_attention, get_self_attn_feat
from .pipeline_utils import _in_step, _classify_blocks
from .pipelines import *

# Take from huggingface/diffusers
class StableDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        nsfw_content_detected (`List[bool]`)
            List indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content or
            `None` if safety checking could not be performed.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


# Take from huggingface/diffusers
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


@register_pipeline('SDXLPipeline')
class SDXLPipeline(StableDiffusionXLPipeline):
    """
    Method adopted from https://github.com/huggingface/diffusers/blob/v0.21.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
    """

    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            denoising_end: Optional[float] = None,
            guidance_scale: float = 5.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            original_size: Optional[Tuple[int, int]] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Optional[Tuple[int, int]] = None,
            negative_original_size: Optional[Tuple[int, int]] = None,
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            negative_target_size: Optional[Tuple[int, int]] = None,

            # Pose2Pose parameters
            config: Optional[Union[Dict[str, Any], omegaconf.DictConfig]] = None,
            inverted_data=None,
    ):
        assert config is not None, "config is required for Pose2Pose pipeline"
        self.input_config = config

        self.unet = prep_unet_attention(self.unet)
        self.unet = prep_unet_conv(self.unet)
        self.load_pca_info()
        self.running_device = 'cuda'
        self.ref_mask_record = None

        # 0. Default height and width to unet
        height = self.img_size[1] or self.unet.config.sample_size * self.vae_scale_factor
        width = self.img_size[0] or self.unet.config.sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # Compute the mapping token relation
        # inversion prompt need to be a list of prompt

        inversion_prompt = self.input_config.data.inversion.prompt
        obj_pairs = self.input_config.sd_config.obj_pairs
        generate_prompt = prompt

        # Prepare guidance configs
        self.guidance_config = config.guidance
        same_latent = config.sd_config.same_latent
        obj_pairs = extract_data(obj_pairs)
        temp_pairs = list()
        for i in range(len(obj_pairs)):
            pair = obj_pairs[i]
            ref = pair['ref']
            gen = pair['gen']
            try:
                ref_id, _ = compute_token_merge_indices(self.tokenizer, inversion_prompt, ref)
            except:
                ref_id = None
                print(f"Cannot find the token id for \"{ref}\" in the inversion prompt \"{inversion_prompt}\"")

            try:
                gen_id, _ = compute_token_merge_indices(self.tokenizer, generate_prompt, gen)
            except:
                gen_id = None
                print(f"Cannot find the token id for \"{gen}\" in the generate prompt \"{generate_prompt}\"")

            if ref_id is not None and gen_id is not None:
                temp_pairs.append({'ref': ref_id, 'gen': gen_id})

        if len(temp_pairs) == 0:
            raise ValueError("Cannot find any token id for the given obj pairs")
        self.record_obj_pairs = temp_pairs
        self.cross_attn_probs: Dict = {'channels': 0, 'probs': None}

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # Copy the latent for the appearance sample
        if same_latent:
            keep_latents = latents
        latents = torch.cat([latents] * 2, dim=0)

        '''
        Compute the ids for control samples and example samples, and appearance samples
        The order of those samples are :
              For classifier guidance: [unconditional, conditional]
              For unconditional/conditional samples, the orders is [example, control, appearance]

        Example sample is the pose condition input, used to provide spatial information
        Control sample is the sample we want to control
        Appearance sample is the sample used to provide texture regularization

        For DDIM inversion method with only one sample, the order is :
            [uncond-control, uncond-appearance, cond-example, cond-control, cond-appearance]

        For Null-text inversion method with only one control sample, the order is :
            [uncond-example, uncond-control, uncond-appearance, cond-example, cond-control, cond-appearance]

        '''
        num_example_sample: int = len(inverted_data['condition_input'])
        num_appearance_sample: int = len(inverted_data['appearance_input']) if 'appearance_input' in inverted_data.keys() and inverted_data[
                                                                                  'appearance_input'] is not None else 0
        num_control_samples: int = batch_size * num_images_per_prompt
        if num_appearance_sample == 0:
            num_appearance_sample = num_control_samples
        total_samples: int = 0
        if config.data.inversion.method == 'DDIM':
            uncond_example_ids: List[int] = list()
            total_samples += 2 * (num_control_samples + num_appearance_sample) + num_example_sample
        else:
            uncond_example_ids: List[int] = np.arange(num_example_sample).tolist()
            total_samples += 2 * (num_control_samples + num_appearance_sample + num_example_sample)

        cond_example_ids: List[int] = (
                np.arange(0, num_example_sample, 1) + (num_control_samples * 2 + len(uncond_example_ids))).tolist()
        cond_control_ids: List[int] = (np.arange(0, num_control_samples, 1) + (cond_example_ids[-1] + 1)).tolist()
        # Currently use the same number of appearance samples as control samples
        cond_appearance_ids: List[int] = (np.arange(0, num_control_samples, 1) + (cond_control_ids[-1] + 1)).tolist()
        example_ids = uncond_example_ids + cond_example_ids
        keep_ids: List[int] = [ids for ids in np.arange(total_samples).tolist() if ids not in example_ids]

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare guidance configs
        self.guidance_config = config.guidance

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 7.1 Apply denoising_end
        if denoising_end is not None and isinstance(denoising_end, float) and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                score = None
                assert do_classifier_free_guidance, "Currently only support classifier free guidance"
                # Process the latent
                step_timestep: int = t.detach().cpu().item()
                assert step_timestep in inverted_data['condition_input'][0][
                    'all_latents'].keys(), f"timestep {step_timestep} not in inverse samples keys"
                data_samples_latent: torch.Tensor = inverted_data['condition_input'][0]['all_latents'][step_timestep]
                data_samples_latent = data_samples_latent.to(device=self.running_device, dtype=prompt_embeds.dtype)

                if i == 0 and same_latent:
                    latents = data_samples_latent.repeat(2, 1, 1, 1)

                if config.data.inversion.method == 'DDIM':
                    if i == 0 and same_latent and config.sd_config.appearnace_same_latent:
                        latents = data_samples_latent.repeat(2, 1, 1, 1)
                    elif i == 0 and same_latent and not config.sd_config.appearnace_same_latent:
                        latents = torch.cat([data_samples_latent, keep_latents], dim=0)
                        print("Latents shape", latents.shape)
                    latent_list: List[torch.Tensor] = [latents, data_samples_latent, latents]
                else:
                    raise NotImplementedError("Currently only support DDIM method")

                # check if appearance_input is in inverted_data
                if 'appearance_input' in inverted_data.keys() and inverted_data['appearance_input'] is not None:
                    appearance_input = inverted_data['appearance_input'][0]['all_latents'][step_timestep].to(
                        device=self.running_device, dtype=prompt_embeds.dtype)
                    # replace the second batch of the last latent in the latent list
                    latent_list[1] = appearance_input
                    latent_list[-1] = appearance_input

                latent_model_input: torch.Tensor = torch.cat(latent_list, dim=0).to('cuda')
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # process the prompt embedding
                if config.data.inversion.method == 'DDIM':
                    ref_prompt_embeds = inverted_data['condition_input'][0]['prompt_embeds'].to('cuda')
                    ref_added_text_embeds = inverted_data['condition_input'][0]['add_text_embeds'].to('cuda')
                    ref_add_time_ids = inverted_data['condition_input'][0]['add_time_id'].to('cuda')

                    step_prompt_embeds_list: List[torch.Tensor] = [prompt_embeds.chunk(2)[0]] * 2 + [
                        ref_prompt_embeds] + [prompt_embeds.chunk(2)[1]] * 2
                    step_add_text_embeds_list: List[torch.Tensor] = [add_text_embeds.chunk(2)[0]] * 2 + [
                        add_text_embeds.chunk(2)[1]] + [add_text_embeds.chunk(2)[1]] * 2
                    step_add_time_ids_list: List[torch.Tensor] = [add_time_ids.chunk(2)[0]] * 2 + [
                        add_time_ids.chunk(2)[1]] + [add_time_ids.chunk(2)[1]] * 2


                else:
                    raise NotImplementedError("Currently only support DDIM method")

                step_prompt_embeds = torch.cat(step_prompt_embeds_list, dim=0).to('cuda')
                step_add_text_embeds = torch.cat(step_add_text_embeds_list, dim=0).to('cuda')
                step_add_time_ids = torch.cat(step_add_time_ids_list, dim=0).to('cuda')

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": step_add_text_embeds, "time_ids": step_add_time_ids}

                require_grad_flag = False
                # Check if the current step is in the guidance step
                if _in_step(self.guidance_config.pca_guidance, i) or _in_step(self.guidance_config.cross_attn, i):
                    require_grad_flag = True
                # Only require grad when need to compute the gradient for guidance
                if require_grad_flag:
                    latent_model_input.requires_grad_(True)
                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=step_prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                else:
                    with torch.no_grad():
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=step_prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                # Compute loss
                loss = 0
                self.cross_seg = None
                if _in_step(self.guidance_config.cross_attn, i):
                    # Compute the Cross-Attention loss
                    self.compute_cross_attn_mask(cond_control_ids, cond_example_ids, cond_appearance_ids)

                if _in_step(self.guidance_config.pca_guidance, i):
                    # Compute the PCA structure and appearance guidance
                    # Set the select feature to key by default
                    try:
                        select_feature = self.guidance_config.pca_guidance.select_feature
                    except:
                        select_feature = "key"

                    if select_feature == 'query' or select_feature == 'key' or select_feature == 'value':
                        pca_loss = self.compute_attn_pca_loss(cond_control_ids, cond_example_ids, cond_appearance_ids,
                                                              i)
                        loss += pca_loss
                    elif select_feature == 'conv':
                        pca_loss = self.compute_conv_pca_loss(cond_control_ids, cond_example_ids, cond_appearance_ids,
                                                              i)
                        loss += pca_loss

                temp_control_ids = None
                if isinstance(loss, torch.Tensor):
                    gradient = torch.autograd.grad(loss, latent_model_input, allow_unused=True)[0]
                    gradient = gradient[cond_control_ids]
                    assert gradient is not None, f"Step {i}: grad is None"
                    score = gradient.detach()
                    temp_control_ids: List[int] = np.arange(num_control_samples).tolist()

                # perform guidance
                if do_classifier_free_guidance:
                    # Remove the example samples
                    noise_pred = noise_pred[keep_ids]
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, score=score,
                                              guidance_scale=self.input_config.sd_config.grad_guidance_scale,
                                              indices=temp_control_ids,
                                              **extra_step_kwargs, return_dict=False)[0].detach()

                score = None
                torch.cuda.empty_cache()
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            with torch.no_grad():
                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
                # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionPipelineOutput(images=image)

    def load_pca_info(self):
        path = self.input_config.sd_config.pca_paths[0]
        self.loaded_pca_info = torch.load(path)

    def _compute_feat_loss(self, feat, pca_info, cond_control_ids, cond_example_ids, cond_appearance_ids, step,
                           reg_included=False, reg_feature=None, ):
        feat_copy = feat if reg_feature is None else reg_feature
        loss: List[torch.Tensor] = []
        # Feat in the shape [bs,h*w,channels]
        feat_mean: torch.Tensor = pca_info['mean'].to(self.running_device)
        feat_basis: torch.Tensor = pca_info['basis'].to(self.running_device)
        n_components: int = self.guidance_config.pca_guidance.structure_guidance.n_components

        # print(feat.shape)
        centered_feat = feat - feat_mean
        # Compute the projection
        feat_proj = torch.matmul(centered_feat, feat_basis.T)

        if self.guidance_config.pca_guidance.structure_guidance.normalized:
            # Normalize the projection by the max and min value
            feat_proj = feat_proj.permute(0, 2, 1)
            feat_proj_max = feat_proj.max(dim=-1, keepdim=True)[0].detach()
            feat_proj_min = feat_proj.min(dim=-1, keepdim=True)[0].detach()
            feat_proj = (feat_proj - feat_proj_min) / (feat_proj_max - feat_proj_min + 1e-7)
            feat_proj = feat_proj.permute(0, 2, 1)
        feat_proj = feat_proj[:, :, :n_components]

        if self.guidance_config.pca_guidance.structure_guidance.mask_tr > 0:
            # Get the activation mask for each component
            # Check the policy for pca guidance
            if self.input_config.data.inversion.policy == 'share':

                ref_feat = feat_proj[cond_example_ids].mean(dim=0, keepdim=True)
                num_control_samples: int = len(cond_control_ids)
                ref_feat = ref_feat.repeat(num_control_samples, 1, 1)
                res = int(math.sqrt(feat_proj.shape[1]))
                # Select the mask for the control samples
                if self.guidance_config.pca_guidance.structure_guidance.mask_type == 'tr':
                    ref_mask = ref_feat > self.guidance_config.pca_guidance.structure_guidance.mask_tr
                elif self.guidance_config.pca_guidance.structure_guidance.mask_type == 'cross_attn':
                    # Currently, only take the first object pair
                    obj_pair = self.record_obj_pairs[0]
                    example_token_ids = obj_pair['ref']
                    example_sample_ids = self.new_id_record[0]
                    example_sample_probs = self.cross_attn_probs['probs'][example_sample_ids]
                    example_token_probs = example_sample_probs[:, example_token_ids].sum(dim=1)
                    # use max, min value to normalize the probs
                    example_token_probs = (example_token_probs - example_token_probs.min(dim=-1, keepdim=True)[0]) / (
                            example_token_probs.max(dim=-1, keepdim=True)[0] -
                            example_token_probs.min(dim=-1, keepdim=True)[0] + 1e-7)
                    mask_res = int(math.sqrt(example_token_probs.shape[1]))
                    if res != mask_res:
                        example_token_probs = example_token_probs.unsqueeze(0).reshape(1, 1, mask_res, mask_res)
                        example_token_probs = F.interpolate(example_token_probs, size=(res, res),
                                                            mode='bicubic').squeeze(1).reshape(1, -1)

                    ref_mask = example_token_probs > self.guidance_config.pca_guidance.structure_guidance.mask_tr
                    ref_mask = ref_mask.to(self.running_device).unsqueeze(-1).repeat(num_control_samples,
                                                                                     1, ref_feat.shape[-1])

                # Compute the loss
                temp_loss: torch.Tensor = F.mse_loss(ref_feat[ref_mask], feat_proj[cond_control_ids][ref_mask])

                # Compute l2 penalty loss
                penalty_factor: float = float(self.guidance_config.pca_guidance.structure_guidance.penalty_factor)
                fliped_mask = ~ref_mask
                if self.guidance_config.pca_guidance.structure_guidance.penalty_type == 'max':
                    # Compute the max value in the fliped_mask
                    score1 = (feat_proj[cond_example_ids] * fliped_mask).max(dim=1, keepdim=True)[0]
                    score2 = F.relu((feat_proj[cond_control_ids] * fliped_mask) - score1)
                    penalty_loss = penalty_factor * F.mse_loss(score2, torch.zeros_like(score2))
                    temp_loss += penalty_loss
                elif self.guidance_config.pca_guidance.structure_guidance.penalty_type == 'hard':
                    # Compute the max value in the fliped_mask
                    score1 = (feat_proj[cond_example_ids] * fliped_mask).max(dim=1, keepdim=True)[0]
                    # assign hard value to the score1
                    score1 = torch.ones_like(score1) * self.guidance_config.pca_guidance.structure_guidance.hard_value
                    score2 = F.relu((feat_proj[cond_control_ids] * fliped_mask) - score1)
                    penalty_loss = penalty_factor * F.mse_loss(score2, torch.zeros_like(score2))
                    temp_loss += penalty_loss
                else:
                    raise NotImplementedError("Only max penalty type has been implemented")
                loss.append(temp_loss)

            else:
                raise NotImplementedError("Only \'share\' policy has been implemented")

        else:
            if self.input_config.data.inversion.policy == 'share':
                ref_feat = feat_proj[cond_example_ids].mean(dim=0, keepdim=True).detach()
                num_control_samples: int = len(cond_control_ids)
                ref_feat = ref_feat.repeat(num_control_samples, 1, 1)
                # Compute the loss
                temp_loss: torch.Tensor = F.mse_loss(ref_feat, feat_proj[cond_control_ids])

            elif self.input_config.data.inversion.policy == 'separate':
                assert len(cond_control_ids) == len(cond_example_ids), \
                    "The number of control samples and example samples must be the same if use separate policy"
                ref_feat = feat_proj[cond_example_ids]
                control_feat = feat_proj[cond_control_ids]
                temp_loss: torch.Tensor = F.mse_loss(ref_feat, control_feat)

            else:
                raise NotImplementedError("Only \'share\' policy has been implemented")

            loss.append(temp_loss)

        # Compute the texture regularization loss
        reg_factor = float(self.guidance_config.pca_guidance.appearance_guidance.reg_factor)
        if reg_included and reg_factor > 0:
            app_n_components = self.guidance_config.pca_guidance.appearance_guidance.app_n_components

            def compute_app_loss(feature, weights, tr, control_ids, appearance_ids):
                """
                Compute the weighted average feature loss based on the given weights and feature

                :return: loss: MSE_loss between two weighted average feature
                """
                weights = weights[:, :, :app_n_components]
                B, C, W = feature.shape
                _, _, K = weights.shape
                mask = (weights > tr).float()
                mask = weights * mask
                expanded_mask = mask.unsqueeze(-2).expand(B, C, W, K)
                masked_feature = feature.unsqueeze(-1) * expanded_mask
                count = mask.sum(dim=1, keepdim=True)
                avg_feature = masked_feature.sum(dim=1) / (count + 1e-5)
                return F.mse_loss(avg_feature[control_ids], avg_feature[appearance_ids].detach())

            temp_loss_list = []
            for temp_feat in feat_copy:
                # Compute the appearance guidance loss
                temp_loss: torch.Tensor = compute_app_loss(temp_feat, feat_proj,
                                                           self.guidance_config.pca_guidance.appearance_guidance.tr,
                                                           cond_control_ids, cond_appearance_ids)
                temp_loss_list.append(temp_loss)
            temp_loss = torch.stack(temp_loss_list).mean()
            loss.append(temp_loss * reg_factor)
        loss = torch.stack(loss).sum()
        return loss

    def compute_conv_pca_loss(self, cond_control_ids, cond_example_ids, cond_appearance_ids, step_i):
        """
        Compute the PCA Conv loss based on the given condition control, example, and appearance IDs.

        This method is not used in FreeControl method, only for ablation study
        :param cond_control_ids:
        :param cond_example_ids:
        :param cond_appearance_ids:
        :param step_i:
        :return:
        """
        # The Conv loss is not used in our method
        # The new tensor follows this order: example, control, appearance
        combined_list = cond_example_ids + cond_control_ids + cond_appearance_ids
        new_cond_example_ids = np.arange(len(cond_example_ids)).tolist()
        new_cond_control_ids = np.arange(len(cond_example_ids), len(cond_control_ids) + len(cond_example_ids)).tolist()
        new_cond_appearance_ids = np.arange(len(cond_control_ids) + len(cond_example_ids), len(combined_list)).tolist()

        step_pca_info: dict = self.loaded_pca_info[step_i]
        total_loss = []
        # get correct blocks
        for block_name in self.guidance_config.pca_guidance.blocks:
            if "up_blocks" in block_name:
                block_id = int(block_name.split(".")[-1])
            else:
                raise NotImplementedError("Only support up_blocks")
            for j in range(len(self.unet.up_blocks[block_id].resnets)):
                name = f"up_blocks.{block_id}.resnets.{j}"
                conv_feat = self.unet.up_blocks[block_id].resnets[j].record_hidden_state
                conv_feat = conv_feat[combined_list]
                conv_feat = conv_feat.permute(0, 2, 3, 1).contiguous().reshape(len(combined_list), -1,
                                                                               conv_feat.shape[1])
                conv_pca_info: dict = step_pca_info['conv'][name]

                loss = self._compute_feat_loss(conv_feat, conv_pca_info, new_cond_control_ids, new_cond_example_ids,
                                               new_cond_appearance_ids, step_i, reg_included=True,
                                               reg_feature=[conv_feat])
                total_loss.append(loss)
        weight = float(self.guidance_config.pca_guidance.weight)
        if self.guidance_config.pca_guidance.warm_up.apply and step_i < self.guidance_config.pca_guidance.warm_up.end_step:
            weight = weight * (step_i / self.guidance_config.pca_guidance.warm_up.end_step)
        loss = torch.stack(total_loss).mean()
        loss = loss * weight
        return loss

    def compute_attn_pca_loss(self, cond_control_ids, cond_example_ids, cond_appearance_ids, step_i):
        """
        Compute the PCA Semantic loss based on the given condition control, example, and appearance IDs.

        This function computes the PCA loss by first creating a combined list of the given IDs, then
        reordering them as example, control, and appearance. The function then retrieves the hidden state
        from the UNet, and subsequently, for each attention module in the UNet, computes the PCA loss
        using the module's key tensor. Finally, the function computes the PCA weight based on the guidance
        config, multiplies the computed PCA loss by this weight, and returns the result.

        Parameters:
        - cond_control_ids (List[int]): List of control condition IDs.
        - cond_example_ids (List[int]): List of example condition IDs.
        - cond_appearance_ids (List[int]): List of appearance condition IDs.
        - step_i (int): The current step in the diffusion process.

        Returns:
        - torch.Tensor: The computed PCA loss.

        """
        # Only save the cond sample, remove the uncond sample to compute loss
        # The new tensor follows this order: example, control, appearance
        combined_list = cond_example_ids + cond_control_ids + cond_appearance_ids
        new_cond_example_ids = np.arange(len(cond_example_ids)).tolist()
        new_cond_control_ids = np.arange(len(cond_example_ids), len(cond_control_ids) + len(cond_example_ids)).tolist()
        new_cond_appearance_ids = np.arange(len(cond_control_ids) + len(cond_example_ids), len(combined_list)).tolist()

        pca_loss = []
        step_pca_info: dict = self.loaded_pca_info[step_i]
        # 1. Loop though all layers to get the query, key, and Compute the PCA loss
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "Attention" and 'attn1' in name and 'attentions' in name and \
                    _classify_blocks(self.guidance_config.pca_guidance.blocks, name):
                try:
                    select_feature = self.guidance_config.pca_guidance.select_feature
                except:
                    select_feature = 'key'

                self.current_step = step_i
                # Compute the PCA loss with selected feature
                if select_feature == 'key':
                    key: torch.Tensor = module.processor.key[combined_list]
                    key_pca_info: dict = step_pca_info['attn_key'][name]
                    module_pca_loss = self._compute_feat_loss(key, key_pca_info, new_cond_control_ids,
                                                              new_cond_example_ids, new_cond_appearance_ids,
                                                              step_i,
                                                              reg_included=True, reg_feature=[key])
                elif select_feature == 'query':
                    query: torch.Tenosr = module.processor.query[combined_list]
                    query_pca_info: dict = step_pca_info['attn_query'][name]
                    module_pca_loss = self._compute_feat_loss(query, query_pca_info, new_cond_control_ids,
                                                              new_cond_example_ids, new_cond_appearance_ids,
                                                              step_i,
                                                              reg_included=True, reg_feature=[query])
                else:
                    value: torch.Tensor = module.processor.value[combined_list]
                    value_pca_info: dict = step_pca_info['attn_value'][name]
                    module_pca_loss = self._compute_feat_loss(value, value_pca_info, new_cond_control_ids,
                                                              new_cond_example_ids, new_cond_appearance_ids,
                                                              step_i,
                                                              reg_included=True, reg_feature=[value])

                pca_loss.append(module_pca_loss)
        # 2. compute pca weight
        weight = float(self.guidance_config.pca_guidance.weight)
        if self.guidance_config.pca_guidance.warm_up.apply and step_i < self.guidance_config.pca_guidance.warm_up.end_step:
            weight = weight * (step_i / self.guidance_config.pca_guidance.warm_up.end_step)

        # 3. compute the PCA loss
        pca_loss = torch.stack(pca_loss).mean() * weight
        return pca_loss

    def compute_cross_attn_mask(self, cond_control_ids, cond_example_ids, cond_appearance_ids):
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__

            combined_list = cond_example_ids + cond_control_ids + cond_appearance_ids
            new_cond_example_ids = np.arange(len(cond_example_ids)).tolist()
            new_cond_control_ids = np.arange(len(cond_example_ids),
                                             len(cond_control_ids) + len(cond_example_ids)).tolist()
            new_cond_appearance_ids = np.arange(len(cond_example_ids) + len(cond_example_ids),
                                                len(cond_control_ids) + len(cond_example_ids) + len(
                                                    cond_appearance_ids)).tolist()

            # Log those new ids
            self.new_id_record: List[List[int]] = [new_cond_example_ids, new_cond_control_ids, new_cond_appearance_ids]

            if module_name == "Attention" and 'attn2' in name and 'attentions' in name and \
                    _classify_blocks(self.input_config.guidance.cross_attn.blocks, name):
                # Combine the condition sample for [example, control, appearance], and compute cross-attention weight
                query = module.processor.query[combined_list]
                key = module.processor.key[combined_list]

                query = module.processor.attn.head_to_batch_dim(query).contiguous()
                key = module.processor.attn.head_to_batch_dim(key).contiguous()
                attention_mask = module.processor.attention_mask
                attention_probs = module.processor.attn.get_attention_scores(query, key, attention_mask)
                source_batch_size = int(attention_probs.shape[0] // len(combined_list))
                # record the attention probs and update the averaged attention probs
                reshaped_attention_probs = attention_probs.detach().reshape(len(combined_list), source_batch_size, -1,
                                                                            77).permute(1, 0, 3, 2)
                channel_num = reshaped_attention_probs.shape[0]
                reshaped_attention_probs = reshaped_attention_probs.mean(dim=0)
                # We followed the method in https://arxiv.org/pdf/2210.04885.pdf to compute the cross-attention mask
                if self.cross_attn_probs['probs'] is None:
                    updated_probs = reshaped_attention_probs
                else:
                    updated_probs = (self.cross_attn_probs['probs'] * self.cross_attn_probs[
                        'channels'] + reshaped_attention_probs * channel_num) / (
                                                self.cross_attn_probs['channels'] + channel_num)
                self.cross_attn_probs['probs'] = updated_probs.detach()
                self.cross_attn_probs['channels'] += channel_num
        return

    @torch.no_grad()
    def invert(self,
               img: Union[List[PIL.Image.Image], PIL.Image.Image] = None,
               inversion_config: omegaconf.dictconfig = None,
               use_cache=True):
        """
        Method adopted from

        Invert the selected image to latent space using the selected method.
        Args:
            img: Image to invert. If None, use the default image from the config file.
            prompt: Prompt to invert. If None, use the default prompt from the config file.
            inversion_config: Config file for inversion. If None, use the default config file.
                default : inversion_config = {
                                    'select_inversion_method': 'DDIM',
                                    'fixed_size': None,
                                    'prompt': None,
                                    'num_inference_steps': 50,
                                }

        Returns:
            img_data: Dict with the following keys
                "prompt": A String of the inversion Prompt
                "all_latents": A list of all the latents from the inversion process
                "img": A Tensor of the processed original image
                "pil_img": A PIL image of the processed original image
                "prompt_embeds": A Tensor of the prompt embeddings
        """

        # TODO: Implement this method to invert image

        select_inversion_method = inversion_config['method']
        assert select_inversion_method in ['DDIM', 'NTI',
                                           'NPI'], "Inversion method not supported, please select from ['DDIM', 'NTI', 'NPI']"

        if select_inversion_method == 'DDIM':
            self.inverse_scheduler = DDIMInverseScheduler.from_config(self.scheduler.config)
            # Check if needed to resize to a fixed size
            if inversion_config.fixed_size is not None:
                img_size = inversion_config.fixed_size
                if isinstance(img, PIL.Image.Image):
                    img = img.resize(img_size)

            if isinstance(img, PIL.Image.Image):
                print("Image size: ", img.size)
                self.img_size: tuple = img.size
            else:
                # TODO: Implement this method with the list of images
                raise NotImplementedError("Inversion with a list of images not supported yet")

            # Check if there is the content
            out_folder = os.path.join(inversion_config.target_folder, select_inversion_method,
                                      inversion_config.sd_model)
            prompt: str = inversion_config.prompt
            # Use the image content and the prompt as key to do the inversion
            img_data = None
            data_key = generate_hash_key(img, prompt=prompt)
            if use_cache:
                img_data = get_data(out_folder, data_key)

            if img_data is not None:
                return img_data

            inv_latents, _, all_latent, prompt_embeds, add_text_embeds, add_time_id = self.ddim_inversion(prompt,
                                                                                                          image=img,
                                                                                                          num_inference_steps=inversion_config.num_inference_steps,
                                                                                                          num_reg_steps=0,
                                                                                                          return_dict=False)
            img_data: Dict = {
                'prompt': prompt,
                'all_latents': all_latent,
                'img': PILtoTensor(img),
                'pil_img': img,
                'prompt_embeds': prompt_embeds,
                'add_text_embeds': add_text_embeds,
                'add_time_id': add_time_id
            }
            if use_cache:
                save_data(img_data, out_folder, data_key)

            return img_data

        else:
            raise NotImplementedError("Inversion method not implemented yet")

        pass

    def prepare_image_latents(self, image, batch_size, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        if image.shape[1] == 4:
            latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if isinstance(generator, list):
                latents = [
                    self.vae.encode(image[i: i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0)
            else:

                needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
                print(needs_upcasting)
                if needs_upcasting:
                    self.upcast_vae()
                    image = image.to(torch.float32)
                    # image = image.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
                latents = self.vae.encode(image).latent_dist.sample(generator)

                # cast back to fp16 if needed
                if needs_upcasting:
                    self.vae.to(dtype=torch.float16)

            latents = self.vae.config.scaling_factor * latents

        if batch_size != latents.shape[0]:
            if batch_size % latents.shape[0] == 0:
                # expand image_latents for batch_size
                deprecation_message = (
                    f"You have passed {batch_size} text prompts (`prompt`), but only {latents.shape[0]} initial"
                    " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                    " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                    " your script to pass as many initial images as text prompts to suppress this warning."
                )
                deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
                additional_latents_per_image = batch_size // latents.shape[0]
                latents = torch.cat([latents] * additional_latents_per_image, dim=0)
            else:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {latents.shape[0]} to {batch_size} text prompts."
                )
        else:
            latents = torch.cat([latents], dim=0)

        return latents

    @torch.no_grad()
    def ddim_inversion(
            self,
            prompt: Optional[str] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            image: Union[
                torch.FloatTensor,
                PIL.Image.Image,
                np.ndarray,
                List[torch.FloatTensor],
                List[PIL.Image.Image],
                List[np.ndarray],
            ] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            cross_attention_guidance_amount: float = 0.1,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,

            denoising_end: Optional[float] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            original_size: Optional[Tuple[int, int]] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Optional[Tuple[int, int]] = None,
            negative_original_size: Optional[Tuple[int, int]] = None,
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            negative_target_size: Optional[Tuple[int, int]] = None,
    ):

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        height = 1024
        width = 1024
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # 3. Preprocess image
        image = self.image_processor.preprocess(image)

        # 4. Prepare latent variables
        latents = self.prepare_image_latents(image, batch_size, self.vae.dtype, device, generator).to(torch.float16)

        # 5. Encode input prompt
        num_images_per_prompt = 1
        # 6. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        self.inverse_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.inverse_scheduler.timesteps

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
        # 7.1 Apply denoising_end
        if denoising_end is not None and isinstance(denoising_end, float) and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        all_latents = {}
        # 7. Denoising loop where we obtain the cross-attention maps.
        num_warmup_steps = len(timesteps) - num_inference_steps * self.inverse_scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                timestep_key = t.detach().cpu().item()
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.inverse_scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text,
                                                   guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.inverse_scheduler.step(noise_pred, t, latents).prev_sample
                assert not torch.isnan(latents).any(), "NaN in latents"

                all_latents[timestep_key] = latents.detach().cpu()

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.inverse_scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        inverted_latents = latents.detach().clone()

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (inverted_latents, None, all_latents, prompt_embeds.detach().cpu(), add_text_embeds.detach().cpu(),
                    add_time_ids.detach().cpu())

        return None


    @torch.no_grad()
    def sample_semantic_bases(self,
                              prompt: Union[str, List[str]] = None,
                              prompt_2: Optional[Union[str, List[str]]] = None,
                              height: Optional[int] = None,
                              width: Optional[int] = None,
                              num_inference_steps: int = 50,
                              denoising_end: Optional[float] = None,
                              guidance_scale: float = 5.0,
                              negative_prompt: Optional[Union[str, List[str]]] = None,
                              negative_prompt_2: Optional[Union[str, List[str]]] = None,
                              num_images_per_prompt: Optional[int] = 1,
                              eta: float = 0.0,
                              generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                              latents: Optional[torch.FloatTensor] = None,
                              prompt_embeds: Optional[torch.FloatTensor] = None,
                              negative_prompt_embeds: Optional[torch.FloatTensor] = None,
                              pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
                              negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
                              output_type: Optional[str] = "pil",
                              return_dict: bool = True,
                              callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
                              callback_steps: int = 1,
                              cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                              guidance_rescale: float = 0.0,
                              original_size: Optional[Tuple[int, int]] = None,
                              crops_coords_top_left: Tuple[int, int] = (0, 0),
                              target_size: Optional[Tuple[int, int]] = None,
                              negative_original_size: Optional[Tuple[int, int]] = None,
                              negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
                              negative_target_size: Optional[Tuple[int, int]] = None,

                              # FreeControl Stage1 parameters
                              num_batch: int = 1,
                              config: omegaconf.dictconfig = None,
                              num_save_basis: int = 64,
                              num_save_steps: int = 120,
                              ):
        """
        The sample_pca_components function to the pipeline to generate semantic bases.

        Args:
            # Default parameters please check call method for more details
            # FreeControl Stage1 parameters
            num_batch: int = 1, The number of batches to generate.
                The number of seed images generated will be num_batch * num_save_basis
            config: omegaconf.dictconfig = None, The config file for the pipeline
            num_save_basis : int = 64, The number of leading PC to save
            num_save_steps: int = 120, The number of steps to save the semantic bases

        """
        # 0. Prepare the UNet
        self.unet = prep_unet_attention(self.unet)
        self.unet = prep_unet_conv(self.unet)
        self.sampling_config: omegaconf.dictconfig = config
        self.pca_info: Dict = dict()

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Use the same height and width for original and target size
        original_size = (height, width)
        target_size = (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        all_latents = self.prepare_latents(
            batch_size * num_images_per_prompt * num_batch,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        latent_list = list(all_latents.chunk(num_batch, dim=0))
        latent_list_copy = latent_list[:]

        seg_maps = dict()
        fixed_size = (int(128), int(128))

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        # 7.1 Apply denoising_end
        if denoising_end is not None and isinstance(denoising_end, float) and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        pbar_steps = min(num_save_steps, num_inference_steps)
        with self.progress_bar(total=pbar_steps * num_batch) as progress_bar:
            for i, t in enumerate(timesteps):
                if i >= num_save_steps:
                    break
                # create dict to store the hidden features
                attn_key_dict = dict()

                for latent_id, latents in enumerate(latent_list):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if do_classifier_free_guidance and guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)
                    latent_list[latent_id] = latents

                    # 8. Post-processing the pca features
                    hidden_state_dict, query_dict, key_dict, value_dict = get_self_attn_feat(self.unet,
                                                                                             self.sampling_config.guidance.pca_guidance)
                    for name in hidden_state_dict.keys():
                        def log_to_dict(feat, selected_dict, name):
                            feat = feat.chunk(2)[1]
                            if name in selected_dict.keys():
                                selected_dict[name].append(feat)
                            else:
                                selected_dict[name] = [feat]

                        log_to_dict(key_dict[name], attn_key_dict, name)

                def apply_pca(feat):
                    with torch.autocast(device_type='cuda', dtype=torch.float32):
                        feat = feat.contiguous().to(torch.float32)
                        # feat shape in [bs,channels,16,16]
                        bs, channels, h, w = feat.shape
                        if feat.ndim == 4:
                            X = feat.permute(0, 2, 3, 1).reshape(-1, channels).to('cuda')
                        else:
                            X = feat.permute(0, 2, 1).reshape(-1, channels).to('cuda')
                        # Computing PCA
                        mean = X.mean(dim=0)
                        tensor_centered = X - mean
                        U, S, V = torch.svd(tensor_centered)
                        n_egv = V.shape[-1]

                        if n_egv > num_save_basis and num_save_basis > 0:
                            V = V[:, :num_save_basis]
                        basis = V.T
                    assert mean.shape[-1] == basis.shape[-1]

                    return {
                        'mean': mean.cpu(),
                        'basis': basis.cpu(),
                    }

                def process_feat_dict(feat_dict):
                    for name in feat_dict.keys():
                        feat_dict[name] = torch.cat(feat_dict[name], dim=0)
                        feat_dict[name] = apply_pca(feat_dict[name])

                # Only process for the first num_save_steps
                if i < num_save_steps:
                    process_feat_dict(attn_key_dict)

                    self.pca_info[i] = {
                        'attn_key': attn_key_dict,
                    }

        return self.pca_info