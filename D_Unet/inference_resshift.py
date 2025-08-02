import os, sys
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from sampler import ResShiftSampler

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default=r"E:\liu\database\paper4\odi360\testing", help="Input path.") # in_path: str, folder or image path for LQ imag
    # parser.add_argument("-i", "--in_path", type=str, default=r"D:\liuleiming\testing\SUN", help="Input path.") # in_path: str, folder or image path for LQ imag
    parser.add_argument("--yaml_name", type=str, default="X4_fisheye", help="Name of the YAML.")
    parser.add_argument("-o", "--out_path", type=str, default="./results/X4_fisheye", help="Output path.") # out_path: str, folder save the results
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--mask_path", type=str, default="", help="Mask path for inpainting.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--bs", type=int, default=1, help="Batch size.")
    args = parser.parse_args()

    return args

def get_configs(args):
    ckpt_dir = Path('./weights')
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()
    yaml_path = f'./configs/{args.yaml_name}.yaml'
    configs = OmegaConf.load(yaml_path)
    ckpt_path = ckpt_dir / f'./{args.yaml_name}/D-UNet.pth'
    vqgan_path = ckpt_dir / f'./{args.yaml_name}/VQ-LAE.ckpt'

    # prepare the checkpoint
    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.sf = args.scale
    configs.autoencoder.ckpt_path = str(vqgan_path)

    # save folder
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)

    return configs

def main():
    args = get_parser()

    configs = get_configs(args)

    resshift_sampler = ResShiftSampler(
            configs,
            sf=args.scale,
            use_amp=True,
            seed=args.seed,
            padding_offset=configs.model.params.get('lq_size', 64),
            )

    resshift_sampler.inference(
            args.in_path,
            args.out_path,
            args.yaml_name,
            bs=args.bs,
            noise_repeat=False
            )

if __name__ == '__main__':
    main()
