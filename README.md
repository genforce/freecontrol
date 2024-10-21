> ## FreeControl: Training-Free Spatial Control of Any Text-to-Image Diffusion Model with Any Condition
> ### [[Paper]](https://arxiv.org/abs/2312.07536) [[Project Page]](https://genforce.github.io/freecontrol/) <br>
> [Sicheng Mo](https://sichengmo.github.io/)<sup>1*</sup>, [Fangzhou Mu](https://pages.cs.wisc.edu/~fmu/)<sup>2*</sup>, 
> [Kuan Heng Lin](https://kuanhenglin.github.io)<sup>1</sup>, [Yanli Liu](https://scholar.google.ca/citations?user=YzXIxCwAAAAJ&hl=en)<sup>3</sup>,
> Bochen Guan<sup>3</sup>, [Yin Li](https://www.biostat.wisc.edu/~yli/)<sup>2</sup>, [Bolei Zhou](https://boleizhou.github.io/)<sup>1</sup> <br>
>      <sup>1</sup> UCLA, <sup>2</sup> University of Wisconsin-Madison, <sup>3</sup> Innopeak Technology, Inc <br>
> <sup>*</sup> Equal contribution <br>
> Computer Vision and Pattern Recognition (CVPR), 2024 <br>
<p align="center">
  <img src="docs/assets/teaser1.jpg" alt="teaser" width="90%" height="90%">
</p>

## Overview

This is the official implementation of FreeControl, a Generative AI algorithm for controllable text-to-image generation using pre-trained Diffusion Models.

## Changelog
* 10/21/2024: Added SDXL pipeline (thanks to @shirleyzhu233).

* 02/19/2024: Initial code release. The paper is accepted to CVPR 2024.

## Getting Started

**Environment Setup**
- We provide a [conda env file](environment.yml) for environment setup. 
```bash
conda env create -f environment.yml
conda activate freecontrol
pip install -U diffusers 
pip install -U gradio
```

**Sample Semantic Bases**
- We provide three sample scripts in the [scripts](scripts) folder (one for each base model) to showcase how to compute target semantic bases.
- You may also download pre-computed bases from [google drive](https://drive.google.com/file/d/1o1BcIBANukeJ2pCG064-eNH9hbQoB24Z/view?usp=sharing). Put them in the [dataset](dataset) folder and launch the gradio demo.


**Gradio demo**
- We provide a graphical user interface (GUI) for users to try out FreeControl. Run the following command to start the demo.
```python
python gradio_app.py
```


## Galley:
We are building a gallery of images generated with FreeControl. You are welcome to share your generated images with us. 
## Contact 
[Sicheng Mo](https://sichengmo.github.io/) (smo3@cs.ucla.edu)

## Reference 

```
@article{mo2023freecontrol,
  title={FreeControl: Training-Free Spatial Control of Any Text-to-Image Diffusion Model with Any Condition},
  author={Mo, Sicheng and Mu, Fangzhou and Lin, Kuan Heng and Liu, Yanli and Guan, Bochen and Li, Yin and Zhou, Bolei},
  journal={arXiv preprint arXiv:2312.07536},
  year={2023}
}
```
