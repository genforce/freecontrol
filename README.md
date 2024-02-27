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

[//]: # (## Overview)

[//]: # ()
[//]: # (This is the official PyTorch implementation of our method for Controllable Generation with pre-trained Diffusion Models.)

[//]: # ()
[//]: # (## Quick Start)

## Getting Started

**Environment Setup**
- We proovide a [conda env file](environment.yml) for environment setup. 
```bash
conda env create -f environment.yml
conda activate freecontrol
```

**Sample Semantic Bases**
- We provide two example file under the [scripts](scripts) folder as an example of how to compute target semantic bases.
- You can also download from [google drive](https://drive.google.com/file/d/1o1BcIBANukeJ2pCG064-eNH9hbQoB24Z/view?usp=sharing) to use our pre-computed bases.
- After downloading the file, you can put it under the [dataset](dataset) folder and use the gradio demo.


**Gradio demo**
- We provide the user interface for testing out method. Ruuning the following commend to start the demo.
```python
python gradio_app.py
```


## Galley:
We are building a gallery generated with FreeControl. You are wellcomed to share your generated images with us. 
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