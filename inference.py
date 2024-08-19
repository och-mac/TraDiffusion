import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, LMSDiscreteScheduler
from my_model import unet_2d_condition
import json
from PIL import Image
from utils import Pharse2idx_tokenizer, setup_logger,compute_ca_loss_masks,masks_to_distances_matrixs,points_to_masks,draw_traces
import hydra
import os
from tqdm import tqdm
import numpy as np
import random




def inference(device, unet, vae, tokenizer, text_encoder, prompt, masks, phrases, cfg, logger):

    logger.info("Inference")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Phrases: {phrases}")


    # Get Object Positions
    # 获取prompt 的下标
    logger.info("Conver Phrases to Object Positions")

    # Encode Classifier Embeddings
    uncond_input = tokenizer(
        [""] * cfg.inference.batch_size, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    pil_images = []
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    input_ids,object_positions = Pharse2idx_tokenizer(prompt, phrases,tokenizer)
    
    cond_embeddings = text_encoder(input_ids.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])


    generator = torch.manual_seed(cfg.inference.rand_seed)  # Seed generator to create the inital latent noise

    latents = torch.randn(
        (cfg.inference.batch_size, 4, 64, 64),
        generator=generator,
    ).to(device)

    #生成一个随机潜在特征

    noise_scheduler = LMSDiscreteScheduler(beta_start=cfg.noise_schedule.beta_start, beta_end=cfg.noise_schedule.beta_end,
                                           beta_schedule=cfg.noise_schedule.beta_schedule, num_train_timesteps=cfg.noise_schedule.num_train_timesteps)

    noise_scheduler.set_timesteps(cfg.inference.timesteps)

    latents = latents * noise_scheduler.init_noise_sigma

    object_masks = masks_to_distances_matrixs(masks)


    #初始化结束
    loss = torch.tensor(10000)

    for index, t in enumerate(tqdm(noise_scheduler.timesteps)):
        iteration = 0
        
        while loss.item() / cfg.inference.loss_scale > cfg.inference.loss_threshold and iteration < cfg.inference.max_iter and index < cfg.inference.max_index_step:
            latents = latents.requires_grad_(True)
            latent_model_input = latents
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down,self_attn_map_integrated_up,self_attn_map_integrated_mid,self_attn_map_integrated_down = \
                unet(latent_model_input, t, encoder_hidden_states=cond_embeddings)
        
            # update latents with guidance
            loss = compute_ca_loss_masks(attn_map_integrated_mid, attn_map_integrated_up, object_masks,
                                   object_positions,cfg.inference.move_rate) * cfg.inference.loss_scale

            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]

            latents = latents - grad_cond * noise_scheduler.sigmas[index] ** 2

            iteration += 1
            
            
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            
            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down,self_attn_map_integrated_up,self_attn_map_integrated_mid,self_attn_map_integrated_down = \
                unet(latent_model_input, t, encoder_hidden_states=text_embeddings)

            noise_pred = noise_pred.sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg.inference.classifier_free_guidance * (noise_pred_text - noise_pred_uncond)

            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            torch.cuda.empty_cache()

    with torch.no_grad():
        logger.info("Decode Image...")
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images.extend([Image.fromarray(image) for image in images])
        return pil_images


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
    unet = unet_2d_condition.UNet2DConditionModel(**unet_config).from_pretrained(cfg.general.model_path, subfolder="unet")
    tokenizer = CLIPTokenizer.from_pretrained(cfg.general.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.general.model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg.general.model_path, subfolder="vae")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet.to(device)
    text_encoder.to(device)
    vae.to(device)
    examples = {"prompt": "The man is walking the dog.",
                "phrases": "man; dog",
                "points": [[[0.2,0.4],[0.2,0.6],[0.2,0.7],[0.2,0.8]],
                           [[0.45, 0.8],[0.5, 0.8],[0.6, 0.8]]],
                'save_path': cfg.general.save_path
                }


    # Prepare the save path
    if not os.path.exists(cfg.general.save_path):
        os.makedirs(cfg.general.save_path)
    logger = setup_logger(cfg.general.save_path, __name__)
    logger.info(cfg)
    masks = points_to_masks(examples['points'])
    
    # Inference
    pil_images = inference(device, unet, vae, tokenizer, text_encoder, examples['prompt'],masks, examples['phrases'], cfg, logger)
    pil_images.append(draw_traces(pil_images[0].copy(),masks, examples['phrases']))

    for i,img in enumerate(pil_images):
        image_path = os.path.join(cfg.general.save_path, 'example_{}.png'.format(i))
        img.save(image_path)

if __name__ == "__main__":
    main()