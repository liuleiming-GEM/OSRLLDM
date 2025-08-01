#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-13 16:59:27

import os, sys, math, random

import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf
from contextlib import nullcontext

from utils import util_net
from utils import util_common

import torch
import torch.distributed as dist

from torchvision.transforms.functional import crop
import torchvision.utils as tvu


def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    tvu.save_image(img, file_directory)

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

class BaseSampler:
    def __init__(
            self,
            configs,
            sf=4,
            use_amp=True,
            chop_size=128,
            chop_stride=128,
            chop_bs=1,
            padding_offset=16,
            seed=10000,
            ):
        '''
        Input:
            configs: config, see the yaml file in folder ./configs/
            sf: int, super-resolution scale
            seed: int, random seed
        '''
        self.configs = configs
        self.sf = sf
        self.seed = seed
        self.use_amp = use_amp
        self.padding_offset = padding_offset

        self.setup_dist()  # setup distributed training: self.num_gpus, self.rank

        self.setup_seed()

        self.build_model()

    def overlapping_grid_indices(self, x_cond, img_size, r=None):
        _, c, h, w = x_cond.shape
        chidi1h = [i for i in range(0, h - img_size + 1, r)]
        chidi1w = [i for i in range(0, w - img_size + 1, r)]
        chidi2h = [x // 2 for x in chidi1h]  # 使用整除
        chidi2w = [x // 2 for x in chidi1w]
        chidi4h = [x // 4 for x in chidi1h]
        chidi4w = [x // 4 for x in chidi1w]
        chidi8h = [x // 8 for x in chidi1h]
        chidi8w = [x // 8 for x in chidi1w]
        return chidi1h, chidi1w, chidi2h, chidi2w, chidi4h, chidi4w, chidi8h, chidi8w

    def setup_seed(self, seed=None):
        seed = self.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_dist(self, gpu_id=None):
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            # if mp.get_start_method(allow_none=True) is None:
                # mp.set_start_method('spawn')
            # rank = int(os.environ['LOCAL_RANK'])
            # torch.cuda.set_device(rank % num_gpus)
            # dist.init_process_group(backend='nccl', init_method='env://')
            rank = 0
            torch.cuda.set_device(rank)

        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def write_log(self, log_str):
        if self.rank == 0:
            print(log_str, flush=True)

    def build_model(self):
        # diffusion model
        log_str = f'Building the diffusion model with length: {self.configs.diffusion.params.steps}...'
        self.write_log(log_str)
        self.base_diffusion = util_common.instantiate_from_config(self.configs.diffusion)
        model = util_common.instantiate_from_config(self.configs.model).cuda()
        ckpt_path =self.configs.model.ckpt_path
        assert ckpt_path is not None
        self.write_log(f'Loading Diffusion model from {ckpt_path}...')
        self.load_model(model, ckpt_path)
        self.freeze_model(model)
        self.model = model.eval()

        # autoencoder model
        if self.configs.autoencoder is not None:
            ckpt_path = self.configs.autoencoder.ckpt_path
            assert ckpt_path is not None
            self.write_log(f'Loading AutoEncoder model from {ckpt_path}...')
            autoencoder = util_common.instantiate_from_config(self.configs.autoencoder).cuda()
            self.load_model(autoencoder, ckpt_path)
            autoencoder.eval()
            self.autoencoder = autoencoder
        else:
            self.autoencoder = None

    def load_model(self, model, ckpt_path=None):
        state = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in state:
            state = state['state_dict']
        util_net.reload_model(model, state)

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False

class ResShiftSampler(BaseSampler):
    def sample_func(self, LR, LR_bic, matrix_list, noise_repeat=False, mask=False):
        '''                        im_lq_tensor,
                        im_lq_bic,
                        Mat_LR_patch,
                        Mat_HR_patch,
        Input:
            y0: n x c x h x w torch tensor, low-quality image, [-1, 1], RGB
            mask: image mask for inpainting
        Output:
            sample: n x c x h x w, torch tensor, [-1, 1], RGB
        '''
        if noise_repeat:
            self.setup_seed()
        results = self.base_diffusion.p_sample_loop(
                LR=LR,
                LR_bic=LR_bic,
                matrix_list=matrix_list,
                model=self.model,
                first_stage_model=self.autoencoder,
                noise=None,
                noise_repeat=noise_repeat,
                clip_denoised=(self.autoencoder is None),
                denoised_fn=None,
                progress=False,
                )    # This has included the decoding for latent space
        return results.clamp_(-1.0, 1.0)

    def inference(self, in_path, out_path, yaml_name, mask_path=None, mask_back=True, bs=1, noise_repeat=False):
        '''
        Inference demo.
        Input:
            in_path: str, folder or image path for LQ image
            out_path: str, folder save the results
            bs: int, default bs=1, bs % num_gpus == 0
            mask_path: image mask for inpainting
        '''
        def _process_per_image(im_LR, im_LR_bic, matrix_list,  mask=None):
            '''
            Input:
                im_lq_tensor: b x c x h x w, torch tensor, [-1, 1], RGB
                mask: image mask for inpainting, [-1, 1], 1 for unknown area
            Output:
                im_sr: h x w x c, numpy array, [0,1], RGB
            '''

            context = torch.cuda.amp.autocast if self.use_amp else nullcontext
            with context():
                im_sr_tensor = self.sample_func(
                        im_LR,
                        im_LR_bic,
                        matrix_list,
                        noise_repeat=noise_repeat,
                        mask=mask,
                        )     # 1 x c x h x w, [-1, 1]

            im_sr_tensor = im_sr_tensor * 0.5 + 0.5
            return im_sr_tensor


        # 生成图像
        in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
        out_path = Path(out_path) if not isinstance(out_path, Path) else out_path

        if self.rank == 0:
            assert in_path.exists()
            if not out_path.exists():
                out_path.mkdir(parents=True)  # 检查输出输入路径是否存在

        if self.num_gpus > 1:
            dist.barrier()

        if in_path.is_dir():
            if mask_path is None:  # 无mask的路径，用于遮挡修复
                data_config = {'type': 'base',
                               'params': {'db_dir': str(in_path),
                                          'transform_type': 'default',
                                          'transform_kwargs': {
                                              'mean': 0.5,
                                              'std': 0.5,
                                              },
                                          'need_path': True,
                                          'recursive': True,
                                          'length': None,
                                          }
                               }
            else:
                data_config = {'type': 'inpainting_val',
                               'params': {'lr_path': str(in_path),
                                          'mask_path': mask_path,
                                          'transform_type': 'default',
                                          'transform_kwargs': {
                                              'mean': 0.5,
                                              'std': 0.5,
                                              },
                                          'need_path': True,
                                          'recursive': True,
                                          'im_exts': ['png', 'jpg', 'jpeg', 'JPEG', 'bmp', 'PNG'],
                                          'length': None,
                                          }
                               }
            if yaml_name == "X2_fisheye":
                from data.X2_fisheye import oditest
            if yaml_name == "X4_fisheye":
                from data.X4_fisheye import oditest
            if yaml_name == "X8_fisheye":
                from data.X8_fisheye import oditest
            if yaml_name == "X2_ERP":
                from data.X2_ERP import oditest
            if yaml_name == "X4_ERP":
                from data.X4_ERP import oditest
            if yaml_name == "X8_ERP":
                from data.X8_ERP import oditest
            if yaml_name == "X4_insta":
                from data.X4_insta import oditest
            if yaml_name == "X4_MIG":
                from data.X4_MIG import oditest
            dataset = oditest(data_config)
            self.write_log(f'Find {len(dataset)} images in {in_path}')
            dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=bs,
                    shuffle=False,
                    drop_last=False,
                    )
            ii = 0
            for data in dataloader:
                micro_batchsize = math.ceil(bs / self.num_gpus)
                ind_start = self.rank * micro_batchsize
                ind_end = ind_start + micro_batchsize
                micro_data = {}
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, list):
                            # 如果是列表，将切片的结果扩展到 micro_data[key] 中
                            micro_data[key] = micro_data.get(key, []) + value[ind_start:ind_end]
                        else:
                            # 对于非列表的情况，直接进行切片
                            micro_data[key] = value[ind_start:ind_end]
                elif isinstance(data, list):
                    # 如果 data 本身是一个列表，则直接对其进行切片
                    micro_data = data[ind_start:ind_end]
                else:
                    raise TypeError("Unsupported data type. Expected dict or list.")


        #         micro_data = micro_data[0]
        #         p_size32 = 32
        #         p_size64 = 64
        #         p_size128 = 128
        #         p_size256 = 256
        #         chidi1h, chidi1w, chidi2h, chidi2w, chidi4h, chidi4w, chidi8h, chidi8w = self.overlapping_grid_indices(micro_data['LR_bic'], img_size=p_size256, r=128)
        #         corners256 = [(i, j) for i in chidi1h for j in chidi1w]
        #         corners128 = [(i, j) for i in chidi2h for j in chidi2w]
        #         corners64 = [(i, j) for i in chidi4h for j in chidi4w]
        #         corners32 = [(i, j) for i in chidi8h for j in chidi8w]
        #         x_grid_mask = torch.zeros(1, 3, 1024, 2048).to('cuda')
        #         for (hi, wi) in corners256:
        #             x_grid_mask[:, :, hi:hi + p_size256, wi:wi + p_size256] += 1
        #         # x_grid_masknp = x_grid_mask.cpu().numpy()
        #         matrix_list = micro_data['matrix_list']
        #         if micro_data['LR'].shape[2] == 512:
        #             LR_patch = torch.cat([crop(micro_data['LR'], hi, wi, p_size128, p_size128) for (hi, wi) in corners128], dim=0)
        #         if micro_data['LR'].shape[2] == 256:
        #             LR_patch = torch.cat([crop(micro_data['LR'], hi, wi, p_size64, p_size64) for (hi, wi) in corners64], dim=0)
        #         # if micro_data['LR'].shape[2] == 128:
        #         #     LR_patch = torch.cat([crop(micro_data['LR'], hi, wi, p_size32, p_size32) for (hi, wi) in corners32], dim=0)
        #         LR_bic_patch = torch.cat([crop(micro_data['LR_bic'], hi, wi, p_size256, p_size256) for (hi, wi) in corners256], dim=0)
        #         Mat_32_patch = torch.cat([crop(matrix_list[0], hi, wi, p_size32, p_size32) for (hi, wi) in corners32], dim=0)
        #         Mat_64_patch = torch.cat([crop(matrix_list[1], hi, wi, p_size64, p_size64) for (hi, wi) in corners64], dim=0)
        #         Mat_128_patch = torch.cat([crop(matrix_list[2], hi, wi, p_size128, p_size128) for (hi, wi) in corners128], dim=0)
        #         Mat_256_patch = torch.cat([crop(matrix_list[3], hi, wi, p_size256, p_size256) for (hi, wi) in corners256], dim=0)
        #         matrix_list_new = [Mat_32_patch, Mat_64_patch, Mat_128_patch, Mat_256_patch]
        #         et_output = torch.zeros_like(x_grid_mask, device=x_grid_mask.device)
        #
        #         manual_batching_size = 8
        #         for i in range(0, len(corners256), manual_batching_size):
        #             processed_matrix_list = [feature_map[i:i + manual_batching_size].cuda() for feature_map in matrix_list_new]
        #             outputs = _process_per_image(LR_patch[i:i + manual_batching_size].cuda(),
        #                                           LR_bic_patch[i:i + manual_batching_size].cuda(),
        #                                           processed_matrix_list,
        #                                           mask=micro_data['mask'].cuda() if 'mask' in micro_data else None,)
        #             for idx, (hi, wi) in enumerate(corners256[i:i + manual_batching_size]):
        #                 et_output[0, :, hi:hi + p_size256, wi:wi + p_size256] += outputs[idx]
        #         x0_t = torch.div(et_output, x_grid_mask)
        #         save_image(x0_t, os.path.join(out_path, f"{ii:03d}.png"))
        #         print(f"{ii:03d}.png")
        #         ii += 1
        #
        # self.write_log(f"Processing done, enjoy the results in {str(out_path)}")


                # real
                micro_data = micro_data[0]
                p_size32 = 32
                p_size64 = 64
                p_size128 = 128
                p_size256 = 256
                chidi1h, chidi1w, chidi2h, chidi2w, chidi4h, chidi4w, chidi8h, chidi8w = self.overlapping_grid_indices(micro_data['LR_bic'], img_size=p_size256, r=256)
                corners256 = [(i, j) for i in chidi1h for j in chidi1w]
                corners128 = [(i, j) for i in chidi2h for j in chidi2w]
                corners64 = [(i, j) for i in chidi4h for j in chidi4w]
                corners32 = [(i, j) for i in chidi8h for j in chidi8w]
                x_grid_mask = torch.zeros(1, 3, 2048, 4096).to('cuda')
                for (hi, wi) in corners256:
                    x_grid_mask[:, :, hi:hi + p_size256, wi:wi + p_size256] += 1
                # x_grid_masknp = x_grid_mask.cpu().numpy()
                matrix_list = micro_data['matrix_list']
                # if micro_data['LR'].shape[2] == 1024:
                #     LR_patch = torch.cat([crop(micro_data['LR'], hi, wi, p_size256, p_size256) for (hi, wi) in corners256], dim=0)
                # if micro_data['LR'].shape[2] == 512:
                # LR_patch = torch.cat([crop(micro_data['LR'], hi, wi, p_size128, p_size128) for (hi, wi) in corners128], dim=0)
                # if micro_data['LR'].shape[2] == 256:
                LR_patch = torch.cat([crop(micro_data['LR'], hi, wi, p_size64, p_size64) for (hi, wi) in corners64], dim=0)
                # if micro_data['LR'].shape[2] == 128:
                #     LR_patch = torch.cat([crop(micro_data['LR'], hi, wi, p_size32, p_size32) for (hi, wi) in corners32], dim=0)
                LR_bic_patch = torch.cat([crop(micro_data['LR_bic'], hi, wi, p_size256, p_size256) for (hi, wi) in corners256], dim=0)
                Mat_32_patch = torch.cat([crop(matrix_list[0], hi, wi, p_size32, p_size32) for (hi, wi) in corners32], dim=0)
                Mat_64_patch = torch.cat([crop(matrix_list[1], hi, wi, p_size64, p_size64) for (hi, wi) in corners64], dim=0)
                Mat_128_patch = torch.cat([crop(matrix_list[2], hi, wi, p_size128, p_size128) for (hi, wi) in corners128], dim=0)
                Mat_256_patch = torch.cat([crop(matrix_list[3], hi, wi, p_size256, p_size256) for (hi, wi) in corners256], dim=0)
                matrix_list_new = [Mat_32_patch, Mat_64_patch, Mat_128_patch, Mat_256_patch]
                et_output = torch.zeros_like(x_grid_mask, device=x_grid_mask.device)

                manual_batching_size = 8
                for i in range(0, len(corners256), manual_batching_size):
                    processed_matrix_list = [
                        feature_map[i:i + manual_batching_size].cuda()
                        for feature_map in matrix_list_new
                    ]
                    outputs =  _process_per_image(LR_patch[i:i + manual_batching_size].cuda(),
                                                  LR_bic_patch[i:i + manual_batching_size].cuda(),
                                                  processed_matrix_list,
                                                  mask=micro_data['mask'].cuda() if 'mask' in micro_data else None,
                    )
                    for idx, (hi, wi) in enumerate(corners256[i:i + manual_batching_size]):
                        et_output[0, :, hi:hi + p_size256, wi:wi + p_size256] += outputs[idx]
                x0_t = torch.div(et_output, x_grid_mask)
                save_image(x0_t, os.path.join(out_path, f"{ii:03d}.png"))
                ii += 1
        self.write_log(f"Processing done, enjoy the results in {str(out_path)}")

if __name__ == '__main__':
    pass

