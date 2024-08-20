import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL
from my_model import unet_2d_condition
import json
from PIL import Image
import hydra
import os
import numpy as np
import random
from gradio.components.image_editor import Brush
import gradio as gr
import datetime
from inference import inference
from utils import concat_images,setup_logger,colors,draw_traces
import cv2

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def main(cfg):
    setup_seed(cfg.inference.rand_seed)
    # build and load model
    with open(cfg.general.unet_config) as f:
        unet_config = json.load(f)
    unet = unet_2d_condition.UNet2DConditionModel(**unet_config).from_pretrained(cfg.general.model_path,
                                                                                 subfolder="unet")
    tokenizer = CLIPTokenizer.from_pretrained(cfg.general.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.general.model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg.general.model_path, subfolder="vae")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet.to(device)
    text_encoder.to(device)
    vae.to(device)

    def progress(img_dict_np, prompt, phrase):
        cfg.general.save_path = './traces_output'+str(datetime.datetime.now())
        if not os.path.exists(cfg.general.save_path):
            os.makedirs(cfg.general.save_path)
        img_np = img_dict_np['composite']
        img_np = cv2.resize(img_np, (512,512), interpolation=cv2.INTER_NEAREST)
        maps = []
        pil = Image.new('RGB', (512, 512), color=(167, 179, 195))
        for i in range(len(colors)):
            non_zero_indices = np.where(np.all(img_np == colors[i], axis=-1))
            if non_zero_indices[0].size == 0:
                continue
            bk = np.zeros((512, 512), dtype=np.uint8) 
            bk[non_zero_indices] = 1
            maps.append(bk)  

            for y,x in zip(non_zero_indices[0],non_zero_indices[1]):
                pil.putpixel((x, y), colors[i]) 

        pil.save(os.path.join(cfg.general.save_path,"traces.jpg"))

        # Prepare examples
        examples = {"prompt": prompt,
                    "phrases": phrase,
                    'save_path': cfg.general.save_path
                    }
        
        # Prepare the save path
        if not os.path.exists(cfg.general.save_path):
            os.makedirs(cfg.general.save_path)
        logger = setup_logger(cfg.general.save_path, __name__)

        logger.info(cfg)
        pil_images = inference(device, unet, vae, tokenizer, text_encoder, examples['prompt'], maps, examples['phrases'], cfg, logger)
        pil_images.append(draw_traces(pil_images[0].copy(),maps, examples['phrases']))

        for i,img in enumerate(pil_images):
            image_path = os.path.join(cfg.general.save_path, 'example_{}.png'.format(i))
            img.save(image_path)
            
        horizontal_concatenated = concat_images(pil_images, examples['prompt'])

        return horizontal_concatenated

    white_image_np = np.zeros((512,512, 3), dtype=np.uint8)*255

    iface = gr.Interface(
        fn=progress,
        inputs=[
            gr.Sketchpad(
                value=white_image_np,
                height = "80%",
                width = "80%",
                type='numpy',
                brush=Brush(colors=[
                    "#90EE90",
                    "#FFA500",  
                    "#FF7F50",  
                    "#FF0000",
                    "#0000FF"
                ], color_mode="selecte",    
                            default_size=3),
                image_mode='RGB'
            ),
            gr.Textbox(label='prompt'),
            gr.Textbox(label='phrase')
        ],
        outputs='image',
        title='Traces-guidance'
    )

    iface.launch()


if __name__ == "__main__":
    main()