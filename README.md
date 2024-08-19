# TraDiffusion

The repo is for the paper TraDiffusion: Trajectory-Based Training-Free Image Generation.

## Environment Setup

You can easily set up a environment according to the following command:
```buildoutcfg
conda create -n traces-guidance python=3.8
conda activate traces-guidance
pip install -r requirements.txt
```

## Quick Start 

We provide an example in `inference,py`. The corresponding information will saved in path `./example_output`.  Detail configuration can be found in the `./conf/base_config.yaml` and `inference.py`. You can quickly use with the following commands:
```buildoutcfg
python inference.py general.save_path=./example_output 
```
We also provide a gradio project that you can quickly use with the following commands:
```buildoutcfg
python inference_gradio.py 
```
Here we provide an example of using a Gradio program.

<<<<<<< HEAD
![example](./example.gif)
=======
"https://github.com/och-mac/TraDiffusion/blob/master/example.mp4"

>>>>>>> 1bdd5db6b2104430dce45cf47b3e7fb55d41d228
