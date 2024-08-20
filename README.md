# TraDiffusion：Trajectory-Based Training-Free Image Generation

The repo is for the paper TraDiffusion: Trajectory-Based Training-Free Image Generation.We also provide a [Quick Start guide](#quick-start) and a [Gradio demo](#gradio-demo) to help you quickly get started with this project.


<p align="center">
  <img src="docs/images/intro.png" width=90% height=auto>
</p>


We present TraDiffusion, a training-free, trajectory-based controllable Text-to-Image (T2I) method. Unlike traditional box- or mask-based approaches, TraDiffusion allows users to guide image generation with mouse trajectories. It utilizes the Distance Awareness energy function to focus generation within the defined trajectory. Our comparisons with traditional methods show that TraDiffusion offers simpler and more natural control. Additionally, it effectively manipulates salient regions, attributes, and relationships within generated images using arbitrary or enhanced trajectories. For more visual examples, please check [here](https://github.com/och-mac/TraDiffusion/blob/master/docs/visual_examples.md).

## Model Overview

<p align="center">
  <img src="docs/images/archi.png" width=40% height=auto>
</p>


TraDiffusion uses a pretrained diffusion model and implements a Distance Awareness energy function combined with trajectories to achieve training-free layout control.

Please check our [paper](https://arxiv.org/abs/2408.09739) for more details.

## Quick Start 

### Environment Setup

You can easily set up a environment according to the following command:
```buildoutcfg
conda create -n traces-guidance python=3.8
conda activate traces-guidance
pip install -r requirements.txt
```

### Inference

We provide an example in `inference,py`. The corresponding information will saved in path `./example_output`.  Detail configuration can be found in the `./conf/base_config.yaml` and `inference.py`. You can quickly use with the following commands:
```buildoutcfg
python inference.py general.save_path=./example_output 
```

### Gradio Demo
We also provide a gradio project that you can quickly use with the following commands:
```buildoutcfg
python inference_gradio.py 
```
Here we provide an example of using a Gradio program.

![](./docs/images/example.gif)

## Citation
If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.

```
@misc{wu2024tradiffusiontrajectorybasedtrainingfreeimage,
      title={TraDiffusion: Trajectory-Based Training-Free Image Generation}, 
      author={Mingrui Wu and Oucheng Huang and Jiayi Ji and Jiale Li and Xinyue Cai and Huafeng Kuang and Jianzhuang Liu and Xiaoshuai Sun and Rongrong Ji},
      year={2024},
      eprint={2408.09739},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.09739}, 
}
```
