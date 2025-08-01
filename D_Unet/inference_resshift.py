#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-03-11 17:17:41

import os, sys
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from sampler import ResShiftSampler

from utils.util_opts import str2bool
from basicsr.utils.download_util import load_file_from_url
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

_STEP = {
    'v1': 15,
    'v2': 15,
    'v3': 4,
    'bicsr': 4,
    'inpaint_imagenet': 4,
    'inpaint_face': 4,
    'faceir': 4,
    }
_LINK = {
    'vqgan': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth',
    'vqgan_face256': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/celeba256_vq_f4_dim3_face.pth',
    'vqgan_face512': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/ffhq512_vq_f8_dim8_face.pth',
    'v1': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s15_v1.pth',
    'v2': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s15_v2.pth',
    'v3': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s4_v3.pth',
    'bicsr': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_bicsrx4_s4.pth',
    'inpaint_imagenet': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_inpainting_imagenet_s4.pth',
    'inpaint_face': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_inpainting_face_s4.pth',
    'faceir': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_faceir_s4.pth',
         }

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default=r"E:\liu\database\ODV\MIG\GT\3", help="Input path.") # in_path: str, folder or image path for LQ imag
    # parser.add_argument("-i", "--in_path", type=str, default=r"D:\liuleiming\testing\SUN", help="Input path.") # in_path: str, folder or image path for LQ imag
    parser.add_argument("--yaml_name", type=str, default="X4_MIG", help="Name of the YAML.")
    parser.add_argument("-o", "--out_path", type=str, default="./results/X4_MIG", help="Output path.") # out_path: str, folder save the results
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--mask_path", type=str, default="", help="Mask path for inpainting.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--bs", type=int, default=1, help="Batch size.")
    parser.add_argument(
            "-v",
            "--version",
            type=str,
            default="v1",
            choices=["v1", "v2", "v3"],
            help="Checkpoint version.",
            )  # 训练模型版本选择
    parser.add_argument(
            "--chop_size",
            type=int,
            default=256,
            choices=[512, 256, 64],
            help="Chopping forward.",
            )
    parser.add_argument(
            "--chop_stride",
            type=int,
            default=-1,
            help="Chopping stride.",
            )  # 步幅类似间隔
    parser.add_argument(
            "--task",
            type=str,
            default="bicsr",
            choices=['realsr', 'bicsr', 'inpaint_imagenet', 'inpaint_face', 'faceir'],
            help="Chopping forward.",
            )  # 任务选择
    args = parser.parse_args()

    return args

def get_configs(args):
    ckpt_dir = Path('./weights')
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()
    yaml_path = f'./configs/{args.yaml_name}.yaml'
    configs = OmegaConf.load(yaml_path)
    ckpt_url = _LINK[args.task]
    ckpt_path = ckpt_dir / f'./X4_FIS/D-UNet.pth'
    vqgan_url = _LINK['vqgan']
    vqgan_path = ckpt_dir / f'./X4_FIS/VQ-LAE.ckpt'

    # prepare the checkpoint  检查路径是否存在
    if not ckpt_path.exists():
         load_file_from_url(
            url=ckpt_url,
            model_dir=ckpt_dir,
            progress=True,
            file_name=ckpt_path.name,
            )
    if not vqgan_path.exists():
         load_file_from_url(
            url=vqgan_url,
            model_dir=ckpt_dir,
            progress=True,
            file_name=vqgan_path.name,
            )

    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.sf = args.scale
    configs.autoencoder.ckpt_path = str(vqgan_path)

    # save folder
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)

    # '''
    # 根据输入的参数 chop_size 和 chop_stride 来确定实际使用的步幅大小 chop_stride。
    # 如果 chop_stride 的值小于 0，则根据 chop_size 和 args.scale 计算出一个默认的步幅大小；
    # 如果 chop_stride 的值大于等于 0，则直接使用输入的值。
    # '''
    # if args.chop_stride < 0:
    #     if args.chop_size == 512:
    #         chop_stride = (512 - 64) * (4 // args.scale)
    #     elif args.chop_size == 256:
    #         chop_stride = (256 - 32) * (4 // args.scale)
    #     elif args.chop_size == 64:
    #         chop_stride = (64 - 16) * (4 // args.scale)
    #     else:
    #         raise ValueError("Chop size must be in [512, 256]")
    # else:
    #     chop_stride = args.chop_stride * (4 // args.scale)
    # args.chop_size *= (4 // args.scale)
    # print(f"Chopping size/stride: {args.chop_size}/{chop_stride}")

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

    # setting mask path for inpainting
    if args.task.startswith('inpaint'):
        assert args.mask_path, 'Please input the mask path for inpainting!'
        mask_path = args.mask_path
    else:
        mask_path = None

    resshift_sampler.inference(
            args.in_path,
            args.out_path,
            args.yaml_name,
            mask_path=mask_path,
            bs=args.bs,
            noise_repeat=False
            )

if __name__ == '__main__':
    main()
