import torch
from omegaconf import OmegaConf
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL
from my_model import unet_2d_condition
import json
from PIL import Image
import hydra
import os
import numpy as np
import random
import random
from gradio.components.image_editor import Brush
import gradio as gr
import numpy as np
from PIL import Image
from PIL import Image
import numpy as np
import datetime
from inference import inference
from utils import concat_images,setup_logger

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
        color = [
             (144, 238, 144),  # 浅绿色
                (255,165,0),
                (255, 127, 80),   # 珊瑚色
                (255,0,0),
                (0,0,255)
        ]

        img_np = img_dict_np['composite']
        background = []
        for i in range(len(color)):
            # 找到所有有色元素的索引
            non_zero_indices = np.where(np.all(img_np == color[i], axis=-1))
            if non_zero_indices[0].size == 0:
                continue
            bk = np.zeros((512, 512), dtype=np.uint8) 
            bk[non_zero_indices] = 1
            background.append(bk)  


        pil = Image.new('RGB', (512, 512), color=(167, 179, 195))
        for i in range(len(color)):
            indices = np.where(np.all(img_np == color[i], axis=-1))

            for y,x in zip(indices[0],indices[1]):
                pil.putpixel((x, y), color[i]) 

        pil.save(os.path.join(cfg.general.save_path,"traces.jpg"))

        # 对 img_prompt 的所有值都除以 512
        examples = {"prompt": prompt,
                    "phrases": phrase,
                    'save_path': cfg.general.save_path
                    }
        # Prepare the save path
        if not os.path.exists(cfg.general.save_path):
            os.makedirs(cfg.general.save_path)
        logger = setup_logger(cfg.general.save_path, __name__)

        logger.info(cfg)
        # Save cfg
        logger.info("save config to {}".format(os.path.join(cfg.general.save_path, 'config.yaml')))
        OmegaConf.save(cfg, os.path.join(cfg.general.save_path, 'config.yaml'))
        distances = []
        for i in background:
            distances.append(i)

        pil_images = inference(device, unet, vae, tokenizer, text_encoder, examples['prompt'], distances, examples['phrases'], cfg, logger)

        for i,image in enumerate(pil_images):
            image.save(os.path.join(cfg.general.save_path, '{}.png'.format(i)))
        horizontal_concatenated = concat_images(pil_images, examples['prompt'])

        return horizontal_concatenated

    # 创建一个256x256的纯白图像
    white_image_np = np.ones((256, 256, 3), dtype=np.uint8) * 255

    iface = gr.Interface(
        fn=progress,
        inputs=[
            gr.Sketchpad(
                value=white_image_np,
                height = "512px",
                width = "512px",
                type='numpy',
                brush=Brush(colors=[
                    "#90EE90",
                    "#FFA500",  # 橙色
                    "#FF7F50",  # 珊瑚色
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