import logging
from pathlib import Path
import pdb
from types import SimpleNamespace

import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from torchvision import utils
from torchvision.transforms.functional import to_tensor

from vgen.edit_mask import get_edit_mask
from vgen.object_mask import automatic_mask
from vgen.flow import get_masked_flow
from ldm.models.diffusion.ddim_with_grad import DDIMSamplerWithGrad
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim_inversion import DDIMInversion
from motion_guidance.losses import FlowLoss


LOG = logging.getLogger('vgen')


def load_model_from_config(config, ckpt, device=None):
    """Load diffusion model from configuration

    Function body copied directly from generate.py and slightly adapted.
    """

    LOG.info('Loading model from %s', ckpt)
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        LOG.info('Global Step: %s', pl_sd['global_step'])
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0:
        LOG.debug('missing keys: %s', m)
    if len(u) > 0:
        LOG.debug('unexpected keys: %s', u)

    model = model.to(device)
    model.eval()

    return model


def generate_video(
    save_dir,
    model,
    src_img_path,
    initial_mask,
    target_points,
    mode,
    guidance_schedule,
    prompt='',
    guidance_energy_settings=None,
    # get_mask_fn=automatic_mask, #NOTE: do we need this to be flexible?
    # get_flow_fn=get_masked_flow,
    device=None,
):
    
    if guidance_energy_settings is None:
        guidance_energy_settings = {}

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=False, parents=True)

    # Prepare DDIM sampler and guidance info
    sampler = DDIMSamplerWithGrad(model)
    inverter = DDIMInversion(model)
    
    torch.set_grad_enabled(False)

    # Prepare prompt embeddings
    uncond_embed = model.module.get_learned_conditioning([''])
    cond_embed = model.module.get_learned_conditioning([prompt])

    # ------------------------------- 1. Load image ------------------------------ #
    src_img = cv2.imread(str(src_img_path / 'start.png'))
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    prev_mask = initial_mask
    
    for frameno, cur_target in enumerate(target_points[1:]):  
        src_img_tensor = to_tensor(src_img)[None] * 2 - 1
        src_img_tensor = src_img_tensor.to(device)
    
        cur_target_flow = get_masked_flow(prev_mask, cur_target, dilate=True)
        
        # ------------------------ 2. Generate cached latents ------------------------ #
        # Get the latent representation of the source image
        src_img_latent = inverter.model.module.get_first_stage_encoding(
            model.module.encode_first_stage(src_img_tensor))
        
        # DDIM configs NOTE: HARD CODED AAAAAAAA
        ddim_steps = 500
        scale = 7.5
        ddim_eta = 0.0
        num_recursive_steps = 10
        clip_grad = 200.0
        guidance_weight = 300.0
        log_freq = 25
        
        # Prepare sample output directory
        sample_save_dir = save_dir / 'frames' / f'{frameno:03d}'
        sample_save_dir.mkdir(exist_ok=False, parents=True)
        
        dummy_operation = SimpleNamespace()
        dummy_operation.save_latents = True
        dummy_operation.folder = sample_save_dir
        
        inverter.sample(
            S=ddim_steps,
            batch_size=1,
            shape=[4, *src_img_latent.shape[2:]],
            operation=dummy_operation,
            conditioning=cond_embed,
            eta=ddim_eta,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uncond_embed,
            start_zt=src_img_latent # input is z0 and it saves all zts, I do love input variable names
        )

        latents = []
        for i in range(ddim_steps):
            latent_path = sample_save_dir / 'latents' / f'zt.{i:05}.pth'
            latents.append(torch.load(latent_path))
            
        # NOTE: Reverse order because for SOME reason zt 0000 does not mean 0 
        #step noised, but rather fully noised in their use of cached latents :)))))
        latents = latents[::-1] 
        cached_latents = torch.stack(latents)
    
        # --------------------------- 3. Generate edit mask -------------------------- #
        edit_mask = torch.from_numpy(
            get_edit_mask(cur_target_flow, output_shape=[4, *src_img_latent.shape[2:]])[None]
            ).to(device)
        
        # -------------------------------- 4. Generate ------------------------------- #
        guidance_energy = FlowLoss(
            target_flow=torch.from_numpy(
                np.moveaxis(cur_target_flow, -1, 0)[None] # B, C, H, W
                ).to(device),
            **guidance_energy_settings,
        ).to(device) # NOTE: From equation 4 from the paper

        sample, start_zt, info = sampler.sample(
            num_ddim_steps=ddim_steps,
            cond_embed=cond_embed,
            uncond_embed=uncond_embed,
            batch_size=1,
            shape=[4, *src_img_latent.shape[2:]],
            CFG_scale=scale,
            eta=ddim_eta,
            src_img=src_img_tensor,
            start_zt=cached_latents[0], 
            guidance_schedule=guidance_schedule,
            cached_latents=cached_latents,
            edit_mask=edit_mask,
            num_recursive_steps=num_recursive_steps,
            clip_grad=clip_grad,
            guidance_weight=guidance_weight,
            log_freq=log_freq,
            results_folder=sample_save_dir,
            guidance_energy=guidance_energy,
        )

        # Decode sampled latent
        sample_img = model.module.decode_first_stage(sample)
        sample_img = torch.clamp((sample_img + 1.0) / 2.0, min=0.0, max=1.0)

        # Save
        utils.save_image(sample_img, sample_save_dir / 'pred.png')
        np.save(sample_save_dir / 'losses.npy', info['losses'])
        np.save(sample_save_dir / 'losses_flow.npy', info['losses_flow'])
        np.save(sample_save_dir / 'losses_color.npy', info['losses_color'])
        np.save(sample_save_dir / 'noise_norms.npy', info['noise_norms'])
        np.save(sample_save_dir / 'guidance_norms.npy', info['guidance_norms'])
        
        # Target flow + edit mask
        np.save(sample_save_dir / 'cur_target_flow.npy', cur_target_flow)
        torch.save(edit_mask, sample_save_dir / 'edit_mask.pth')
        
        torch.save(start_zt, sample_save_dir / 'start_zt.pth')
        
        # ------------------------------- 5. Update image and mask ------------------------------ #
        src_img = cv2.imread(str(sample_save_dir / 'pred.png'))
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

        mask_flow = cur_target_flow if mode == "rotate" else None 
        prev_mask = automatic_mask(src_img, 
                                   prev_mask, 
                                   cur_target, 
                                   mode=mode, 
                                   flow=mask_flow)

def main():
    import argparse
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument('input_dir', metavar='INPUT_DIR', type=Path, help='location of src img, pregenerated target points and initial object mask')
    parser.add_argument('output_dir', metavar='OUTPUT_DIR', type=Path, help='path to save results')

    # Diffusion model setup
    parser.add_argument('--model-config', metavar='PATH', type=Path, default='configs/stable-diffusion/v1-inference.yaml', help='path to config which constructs model')
    parser.add_argument('--model-ckpt', metavar='PATH', type=Path, default='chkpts/sd-v1-4.ckpt', help='path to checkpoint of model')

    # Diffusion parameters
    parser.add_argument('--prompt', default='')

    # Guidance parameters
    # parser.add_argument('--edit-mask', metavar='PATH', type=Path, default=None, help='path to edit mask')
    parser.add_argument('--guidance-schedule', metavar='PATH', type=Path, default='data/guidance_schedule.npy', help='use a custom guidance schedule')
    # parser.add_argument('--target-flow', metavar='PATH', type=Path, default='target.pth', help='path to target flow')
    parser.add_argument('--color-weight', type=float, default=100.0)
    parser.add_argument('--flow-weight', type=float, default=3.0)

    # General parameters (low priority)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    
    # Extra
    parser.add_argument('--mode', type=str, choices=['translate', 'rotate'], help='flow mode (translate/rotate)')

    args = parser.parse_args()

    # Set up logging
    if args.debug:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO
    logging.basicConfig(format='%(name)s:%(message)s', level=log_level)

    # Choose device for PyTorch
    if torch.cuda.is_available():
        LOG.info('Running on CUDA device')
        device = torch.device('cuda')
    else:
        LOG.info('Running on CPU')
        device = torch.device('cpu')

    input_dir = args.input_dir

    # Create root directory for pipeline output
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=False, parents=True)

    # Save configuration
    torch.save(args, output_dir / 'config.pth')

    # Prepare diffusion model
    model_config = OmegaConf.load(args.model_config)
    model = load_model_from_config(model_config, args.model_ckpt, device=device)
    model = model.to(device)
    model = torch.nn.DataParallel(
        model,
        device_ids=range(torch.cuda.device_count()),
    )
    model.eval()
    
    # Load initial object mask
    initial_mask = np.load(input_dir / 'initial_mask.npy')
    initial_mask = cv2.dilate(initial_mask.astype('uint8'), np.ones((5, 5), np.uint8), iterations=5).astype(bool)

    mode = args.mode
    
    # Prepare guidance schedule
    guidance_schedule = np.load(args.guidance_schedule)

    # Target *flows* for now, but once we have the code for getting masks and
    # flows, the target points should (apparently?) instead be translation
    # vectors? Or?
    target_points = np.load(input_dir / 'target_points.npy')
    
    # Prepare flow loss
    guidance_energy_settings = {
        'color_weight': args.color_weight,
        'flow_weight': args.flow_weight,
        'oracle': False,
        'occlusion_masking': True,
    }
    
    generate_video(
        output_dir / 'generate_video',
        model,
        input_dir,
        initial_mask,
        target_points,
        mode,
        guidance_schedule,
        prompt=args.prompt,
        guidance_energy_settings=guidance_energy_settings,
        device=device,
    )


if __name__ == '__main__':
    main()
