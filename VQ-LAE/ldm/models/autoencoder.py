import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from packaging import version
from ldm.modules.ema import LitEma
from contextlib import contextmanager
import os
from ldm.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.models.VQGAN import *
from ldm.util import instantiate_from_config
import torchvision.utils as tvu
from ldm.models.LAnet_AE import LAnet

def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    tvu.save_image(img, file_directory)

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

class VQLnet(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 LAnetconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()
        self.embed_dim = embed_dim # 3
        self.n_embed = n_embed # 8192
        self.image_key = image_key # 'image'
        # 优化
        self.scale = LAnetconfig.upscale
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.LAnet = LAnet(**LAnetconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def random_crops_tensor(self, img_tensor, y, x, h, w):
        new_crop = img_tensor[:, :, :, y:y + h, x:x + w]  # 先行在列# 根据位置随机裁剪
        return new_crop

    def overlapping_grid_indices(self, x_cond, img_size, r=None):
        _, c, h, w = x_cond.shape
        cood_hr_h = [i for i in range(0, h - img_size + 1, r)]
        cood_hr_w = [i for i in range(0, w - img_size + 1, r)]
        return cood_hr_h, cood_hr_w

    def encode(self, HR, matrix_list):
        h = self.encoder(HR, matrix_list)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant, LR, matrix_list):
        quant = self.post_quant_conv(quant)
        if self.scale == 2:
            x1, x2 = self.LAnet(LR, matrix_list[2], self.scale)
            dec = self.decoder(z=quant, cond=matrix_list, x2=x1, x4=x2)
            return dec

        elif self.scale == 4:
            x1, x2, x4 = self.LAnet(LR, matrix_list[1], self.scale)
            dec = self.decoder(z=quant, cond=matrix_list, x1=x1, x2=x2, x4=x4)
            return dec

        elif self.scale == 8:
            x1, x2, x4 = self.LAnet(LR, matrix_list[0], self.scale)
            dec = self.decoder(z=quant, cond=matrix_list, x1=x1, x2=x2, x4=x4)
            return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, HR, LR, matrix_list):
        quant, diff, _ = self.encode(HR, matrix_list)
        dec = self.decode(quant, LR, matrix_list)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if isinstance(x, list):
            x = [item[..., None] if len(item.shape) == 3 else item for item in x]
            x = [item.to(memory_format=torch.contiguous_format).float() for item in x]
        else:
            if len(x.shape) == 3:
                x = x[..., None]
            x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        HR = self.get_input(batch, 'HR')
        LR = self.get_input(batch, 'LR')
        matrix_list = self.get_input(batch, 'matrix_list')
        xrec, qloss = self(HR, LR, matrix_list)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, HR, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, HR, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        # with self.ema_scope():
        #     log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        with torch.no_grad():
            HR = self.get_input(batch, 'HR')
            LR = self.get_input(batch, 'LR')
            matrix_list = self.get_input(batch, 'matrix_list')
            img_id = batch['img_id']
            img_id = img_id[0]
            qloss_list = []
            manual_batching_size = 2
            x_grid_mask = torch.zeros(1, 3, 1024, 2048).to('cuda')
            et_output = torch.zeros_like(x_grid_mask, device=x_grid_mask.device)
            gt_output = torch.zeros_like(x_grid_mask, device=x_grid_mask.device)
            cood_hr_h, cood_hr_w = self.overlapping_grid_indices(x_grid_mask, img_size=256, r=256)
            cornershr = [(i, j) for i in cood_hr_h for j in cood_hr_w]
            for (hi, wi) in cornershr:
                x_grid_mask[:, :, hi:hi + 256, wi:wi + 256] += 1
            for i in range(0, len(HR), manual_batching_size):
                xrec, qloss = self(HR[i:i + manual_batching_size].cuda(),
                                   LR[i:i + manual_batching_size].cuda(),
                                   [matrix[i:i + manual_batching_size].cuda() for matrix in matrix_list])
                qloss_list.append(qloss)
                for idx, (hi, wi) in enumerate(cornershr[i:i + manual_batching_size]):
                    et_output[0, :, hi:hi + 256, wi:wi + 256] += xrec[idx]
            for i in range(0, len(HR), manual_batching_size):
                gt = HR[i:i + manual_batching_size].cuda()
                for idx, (hi, wi) in enumerate(cornershr[i:i + manual_batching_size]):
                    gt_output[0, :, hi:hi + 256, wi:wi + 256] += gt[idx]
            qloss_all = torch.stack(qloss_list)
            qloss = qloss_all.mean()
            print('step', self.global_step)
            pred_img = inverse_data_transform(et_output)
            gt_img = inverse_data_transform(gt_output)
            # image_folder = r'./result/X8_ERP'
            image_folder = r'./result/X4_FIS'
            psnr = self.calculate_psnr(pred_img, gt_img)
            print('val_psnr:', psnr)
            save_image(pred_img, os.path.join(image_folder, f"{str(self.global_step)}/", f"{img_id}.png"))
            aeloss, log_dict_ae = self.loss(qloss, gt_img, pred_img, 0,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val" + suffix,
                                            )
            discloss, log_dict_disc = self.loss(qloss, gt_img, pred_img, 1,
                                                self.global_step,
                                                last_layer=self.get_last_layer(),
                                                split="val" + suffix,
                                                )
            rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
            self.log(f"val{suffix}/rec_loss", rec_loss,
                       prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"val{suffix}/aeloss", aeloss,
                       prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            if version.parse(pl.__version__) >= version.parse('1.4.0'):
                del log_dict_ae[f"val{suffix}/rec_loss"]
            self.log_dict(log_dict_ae)
            self.log_dict(log_dict_disc)
            torch.cuda.empty_cache()
            return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.LAnet.parameters()) +
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        HR = self.get_input(batch, 'HR')
        LR = self.get_input(batch, 'LR')
        matrix_list = self.get_input(batch, 'matrix_list')

        x = HR.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(HR, LR, matrix_list)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(HR, LR, matrix_list)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

    def calculate_psnr(self, img1, img2):

        img1 = (img1 * 255.).cpu().numpy()
        img2 = (img2 * 255.).cpu().numpy()
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        psnr_sum = 0.0
        num_images = img1.shape[0]

        for i in range(img1.shape[0]):
            mse = np.mean((img1[i] - img2[i]) ** 2)
            if mse == 0:
                psnr = float('inf')
            else:
                psnr = 10. * np.log10(255. * 255. / mse)
            psnr_sum += psnr

        average_psnr = psnr_sum / num_images
        return average_psnr


class VQLnetInterface(VQLnet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def encode(self, x, matrix_list):
        h = self.encoder(x, matrix_list)
        h = self.quant_conv(h)
        return h

    def decode(self, z0_pred, LR, matrix_list):
        quant, emb_loss, info = self.quantize(z0_pred)
        quant = self.post_quant_conv(quant)
        if self.scale == 2:
            x1, x2 = self.LAnet(LR, matrix_list[2], self.scale)
            dec = self.decoder(z=quant, cond=matrix_list, x2=x1, x4=x2)
            return dec

        elif self.scale == 4:
            x1, x2, x4 = self.LAnet(LR, matrix_list[1], self.scale)
            dec = self.decoder(z=quant, cond=matrix_list, x1=x1, x2=x2, x4=x4)
            return dec

        elif self.scale == 8:
            x1, x2, x4 = self.LAnet(LR, matrix_list[0], self.scale)
            dec = self.decoder(z=quant, cond=matrix_list, x1=x1, x2=x2, x4=x4)
            return dec