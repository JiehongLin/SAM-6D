import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
from functools import partial
import timm.models.vision_transformer
from model_utils import (
    LayerNorm2d,
    interpolate_pos_embed,
    get_chosen_pixel_feats,
    sample_pts_feats
)



class ViT(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, **kwargs):
        super(ViT, self).__init__(**kwargs)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)

        out = []
        d = len(self.blocks)
        n = d // 4
        idx_nblock = [d-1, d-n-1, d-2*n-1, d-3*n-1]

        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx in idx_nblock:
                out.append(self.norm(x))
        return out



class ViT_AE(nn.Module):
    def __init__(self, cfg,) -> None:
        super(ViT_AE, self).__init__()
        self.cfg = cfg
        self.vit_type = cfg.vit_type
        self.up_type = cfg.up_type
        self.embed_dim = cfg.embed_dim
        self.out_dim = cfg.out_dim
        self.use_pyramid_feat = cfg.use_pyramid_feat
        self.pretrained = cfg.pretrained

        if self.vit_type == 'vit_base':
            self.vit = ViT(
                    patch_size=16, embed_dim=self.embed_dim, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),)
        elif self.vit_type == 'vit_large':
            self.vit = ViT(
                    patch_size=16, embed_dim=self.embed_dim, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), )
        else:
            assert False

        if self.use_pyramid_feat:
            nblock = 4
        else:
            nblock = 1

        if self.up_type == 'linear':
            self.output_upscaling = nn.Linear(self.embed_dim * nblock, 16 * self.out_dim, bias=True)
        elif self.up_type == 'deconv':
            self.output_upscaling = nn.Sequential(
                nn.ConvTranspose2d(self.embed_dim * nblock, self.out_dim*2, kernel_size=2, stride=2),
                LayerNorm2d(self.out_dim*2),
                nn.GELU(),
                nn.ConvTranspose2d(self.out_dim*2, self.out_dim, kernel_size=2, stride=2),
            )
        else:
            assert False

        if self.pretrained:
            vit_checkpoint = os.path.join('checkpoints', 'mae_pretrain_'+ self.vit_type +'.pth')
            if not os.path.isdir(vit_checkpoint):
                if not os.path.isdir('checkpoints'):
                    os.makedirs('checkpoints')
                model_zoo.load_url('https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_'+ self.vit_type +'.pth', 'checkpoints')

            checkpoint = torch.load(vit_checkpoint, map_location='cpu')
            print("load pre-trained checkpoint from: %s" % vit_checkpoint)
            checkpoint_model = checkpoint['model']
            state_dict = self.vit.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            # interpolate position embedding
            interpolate_pos_embed(self.vit, checkpoint_model)
            msg = self.vit.load_state_dict(checkpoint_model, strict=False)


    def forward(self, x):
        B,_,H,W = x.size()
        vit_outs = self.vit(x)
        cls_tokens = vit_outs[-1][:,0,:].contiguous()
        vit_outs = [l[:,1:,:].contiguous() for l in vit_outs]

        if self.use_pyramid_feat:
            x = torch.cat(vit_outs, dim=2)
        else:
            x = vit_outs[-1]

        if self.up_type == 'linear':
            x = self.output_upscaling(x).reshape(B,14,14,4,4,self.out_dim).permute(0,5,1,3,2,4).contiguous()
            x = x.reshape(B,-1,56,56)
            x = F.interpolate(x, (H,W), mode="bilinear", align_corners=False)
        elif self.up_type == 'deconv':
            x = x.transpose(1,2).reshape(B,-1,14,14)
            x = self.output_upscaling(x)
            x = F.interpolate(x, (H,W), mode="bilinear", align_corners=False)
        return x, cls_tokens




class ViTEncoder(nn.Module):
    def __init__(self, cfg, npoint=2048):
        super(ViTEncoder, self).__init__()
        self.npoint = npoint
        self.rgb_net = ViT_AE(cfg)

    def forward(self, end_points):
        rgb = end_points['rgb']
        rgb_choose = end_points['rgb_choose']
        dense_fm = self.get_img_feats(rgb, rgb_choose)
        dense_pm = end_points['pts']
        assert rgb_choose.size(1) == self.npoint

        if not self.training and 'dense_po' in end_points.keys() and 'dense_fo' in end_points.keys():
            dense_po = end_points['dense_po'].clone()
            dense_fo = end_points['dense_fo'].clone()

            # normalize point clouds
            radius = torch.norm(dense_po, dim=2).max(1)[0]
            dense_pm = dense_pm / (radius.reshape(-1, 1, 1) + 1e-6)
            dense_po = dense_po / (radius.reshape(-1, 1, 1) + 1e-6)

        else:
            tem1_rgb = end_points['tem1_rgb']
            tem1_choose = end_points['tem1_choose']
            tem1_pts = end_points['tem1_pts']
            tem2_rgb = end_points['tem2_rgb']
            tem2_choose = end_points['tem2_choose']
            tem2_pts = end_points['tem2_pts']

            # normalize point clouds
            dense_po = torch.cat([tem1_pts, tem2_pts], dim=1)
            radius = torch.norm(dense_po, dim=2).max(1)[0]
            dense_pm = dense_pm / (radius.reshape(-1, 1, 1) + 1e-6)
            tem1_pts = tem1_pts / (radius.reshape(-1, 1, 1) + 1e-6)
            tem2_pts = tem2_pts / (radius.reshape(-1, 1, 1) + 1e-6)

            dense_po, dense_fo = self.get_obj_feats(
                [tem1_rgb, tem2_rgb],
                [tem1_pts, tem2_pts],
                [tem1_choose, tem2_choose]
            )

        return dense_pm, dense_fm, dense_po, dense_fo, radius

    def get_img_feats(self, img, choose):
        return get_chosen_pixel_feats(self.rgb_net(img)[0], choose)

    def get_obj_feats(self, tem_rgb_list, tem_pts_list, tem_choose_list, npoint=None):
        if npoint is None:
            npoint = self.npoint

        tem_feat_list =[]
        for tem, tem_choose in zip(tem_rgb_list, tem_choose_list):
            tem_feat_list.append(self.get_img_feats(tem, tem_choose))

        tem_pts = torch.cat(tem_pts_list, dim=1)
        tem_feat = torch.cat(tem_feat_list, dim=1)

        return sample_pts_feats(tem_pts, tem_feat, npoint)



