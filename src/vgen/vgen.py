import logging
from pathlib import Path


import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch
from torchvision import utils
from torchvision.transforms.functional import to_tensor


from ldm.models.diffusion.ddim_with_grad import DDIMSamplerWithGrad
from ldm.util import instantiate_from_config
from losses import FlowLoss


LOG = logging.getLogger('vgen')


NUM_FRAMES = 3


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
    src_img,
    guidance_schedule,
    guidance_energy,
    start_zt=None,
    edit_mask=None,
    cached_latents=None,
    prompt='',
):

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=False, parents=True)

    # Prepare DDIM sampler and guidance info
    sampler = DDIMSamplerWithGrad(model)

    torch.set_grad_enabled(False)

    if edit_mask is None:
        edit_mask = torch.zeros(1,4,64,64).bool()

    # Prepare prompt embeddings
    uncond_embed = model.module.get_learned_conditioning([''])
    cond_embed = model.module.get_learned_conditioning([prompt])

    # Prepare sample output directory
    sample_save_dir = save_dir / 'sample'
    sample_save_dir.mkdir(exist_ok=False, parents=True)

    # Sample
    ddim_steps = 500
    scale = 7.5
    ddim_eta = 0.0
    num_recursive_steps = 10
    clip_grad = 200.0
    guidance_weight = 300.0
    log_freq = 5
    sample, start_zt, info = sampler.sample(
        num_ddim_steps=ddim_steps,
        cond_embed=cond_embed,
        uncond_embed=uncond_embed,
        batch_size=1,
        shape=[4, 64, 64],
        CFG_scale=scale,
        eta=ddim_eta,
        src_img=src_img,
        start_zt=start_zt,
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
    torch.save(start_zt, sample_save_dir / 'start_zt.pth')


def load_latents(path):
    path = Path(path)
    latents = []
    for i in range(500):
        latents.append(torch.load(path / f'zt.{i:05}.pth'))
    return torch.stack(latents)


def main():
    import argparse
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument('input_dir', metavar='INPUT_DIR', type=Path, help='location of src img, flows, etc.')
    parser.add_argument('output_dir', metavar='OUTPUT_DIR', type=Path, help='path to save results')

    # Diffusion model setup
    parser.add_argument('--model-config', metavar='PATH', type=Path, default='configs/stable-diffusion/v1-inference.yaml', help='path to config which constructs model')
    parser.add_argument('--model-ckpt', metavar='PATH', type=Path, default='chkpts/sd-v1-4.ckpt', help='path to checkpoint of model')

    # Diffusion parameters
    parser.add_argument('--prompt', default='')

    # Guidance parameters
    parser.add_argument('--edit-mask', metavar='PATH', type=Path, default=None, help='path to edit mask')
    parser.add_argument('--guidance-schedule', metavar='PATH', type=Path, default='data/guidance_schedule.npy', help='use a custom guidance schedule')
    parser.add_argument('--target-flow', metavar='PATH', type=Path, default='target.pth', help='path to target flow')
    parser.add_argument('--color-weight', type=float, default=100.0)
    parser.add_argument('--flow-weight', type=float, default=3.0)

    # General parameters (low priority)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--quiet', action='store_true')

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

    # Load source image
    src_img = to_tensor(Image.open(input_dir / 'pred.png'))[None] * 2 - 1
    src_img = src_img.to(device)

    # Prepare initial noise
    start_zt = torch.load(input_dir / 'start_zt.pth')

    # Prepare edit mask
    if args.edit_mask is not None:
        edit_mask = torch.load(input_dir / 'flows' / args.edit_mask)
    else:
        edit_mask = None

    # Prepare guidance schedule
    guidance_schedule = np.load(args.guidance_schedule)

    # Prepare latents
    cached_latents = load_latents(input_dir / 'latents')

    # Prepare flow loss
    # TODO: discretize target_flow into multiple steps
    target_flow = torch.load(input_dir / 'flows' / args.target_flow)
    guidance_energy = FlowLoss(
        args.color_weight,
        args.flow_weight,
        oracle=False,
        target_flow=target_flow,
        occlusion_masking=True,
    ).to(device)

    generate_video(
        output_dir / 'generate_video',
        model,
        src_img,
        guidance_schedule,
        guidance_energy,
        start_zt=start_zt,
        edit_mask=edit_mask,
        cached_latents=cached_latents,
        prompt=args.prompt,
    )


if __name__ == '__main__':
    main()
