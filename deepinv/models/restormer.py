# Restormer: Efficient Transformer for High-Resolution Image Restoration
# Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
# https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from .utils import get_weights_url
from einops import rearrange
import os


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


##########################################################################
# Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
# Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=self.bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=self.bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=self.bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
# Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=self.bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=self.bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=self.bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=self.bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
# Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
# ---------- Restormer -----------------------


class Restormer(nn.Module):

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 dim=48,
                 mode="denoising",
                 pretrained="download",
                 device=None,
                 train=False,
                 **kwargs
                 ):
        """Restormer: Efficient Transformer for High-Resolution Image Restoration

        :param in_channels: number of channels of the input, defaults to 3
        :type in_channels: int, optional
        :param out_channels: number of channel of the output, defaults to 3
        :type out_channels: int, optional
        :param dim: dimension of the embeddings, defaults to 48
        :type dim: int, optional
        :param mode: mode of restormer, defaults to "denoising"
        :type mode: str, optional
        :param pretrained: download the model from HFhub or load the path given, defaults to "download"
        :type pretrained: str, optional
        :param device: gpu or cpu, defaults to None
        :type device: str, optional
        :param train: training or testing mode, defaults to False
        :type train: bool, optional
        :raises ValueError: Raised when an invalid model name is provided
        """
        super(Restormer, self).__init__()

        # hard set the model parameter for the least used hyperparameters
        self.num_blocks = kwargs.get("num_blocks", [4, 6, 6, 8])
        self.num_refinement_blocks = kwargs.get("num_refinement_blocks", 4)
        self.heads = kwargs.get("heads", [1, 2, 4, 8])
        self.ffn_expansion_factor = kwargs.get("ffn_expansion_factor", 2.66)
        self.bias = kwargs.get("bias", False)
        self.LayerNorm_type = kwargs.get(
            "LayerNorm_type", 'WithBias')  # Other option 'BiasFree'
        # True for dual-pixel defocus deblurring only. Also set inp_channels=6
        self.dual_pixel_task = kwargs.get("dual_pixel_task", False)
        # color, gray or real for gaussian_denoising
        self.denoising_mode = kwargs.get("denoising_mode", "color")
        # color, gray or real for gaussian_denoising
        self.denoising_sigma = kwargs.get("denoising_sigma", None)

        # model definition
        self.patch_embed = OverlapPatchEmbed(in_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=self.heads[0], ffn_expansion_factor=self.ffn_expansion_factor,
                                            bias=self.bias, LayerNorm_type=self.LayerNorm_type) for i in range(self.num_blocks[0])])

        self.down1_2 = Downsample(dim)  # From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=self.heads[1], ffn_expansion_factor=self.ffn_expansion_factor,
                                            bias=self.bias, LayerNorm_type=self.LayerNorm_type) for i in range(self.num_blocks[1])])

        self.down2_3 = Downsample(int(dim*2**1))  # From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=self.heads[2], ffn_expansion_factor=self.ffn_expansion_factor,
                                            bias=self.bias, LayerNorm_type=self.LayerNorm_type) for i in range(self.num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2))  # From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=self.heads[3], ffn_expansion_factor=self.ffn_expansion_factor,
                                    bias=self.bias, LayerNorm_type=self.LayerNorm_type) for i in range(self.num_blocks[3])])

        self.up4_3 = Upsample(int(dim*2**3))  # From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim*2**3), int(dim*2**2), kernel_size=1, bias=self.bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=self.heads[2], ffn_expansion_factor=self.ffn_expansion_factor,
                                            bias=self.bias, LayerNorm_type=self.LayerNorm_type) for i in range(self.num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2))  # From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim*2**2), int(dim*2**1), kernel_size=1, bias=self.bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=self.heads[1], ffn_expansion_factor=self.ffn_expansion_factor,
                                            bias=self.bias, LayerNorm_type=self.LayerNorm_type) for i in range(self.num_blocks[1])])

        # From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.up2_1 = Upsample(int(dim*2**1))

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=self.heads[0], ffn_expansion_factor=self.ffn_expansion_factor,
                                            bias=self.bias, LayerNorm_type=self.LayerNorm_type) for i in range(self.num_blocks[0])])

        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=self.heads[0], ffn_expansion_factor=self.ffn_expansion_factor,
                                        bias=self.bias, LayerNorm_type=self.LayerNorm_type) for i in range(self.num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = self.dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(
                dim, int(dim*2**1), kernel_size=1, bias=self.bias)
        ###########################

        self.output = nn.Conv2d(int(dim*2**1), out_channels,
                                kernel_size=3, stride=1, padding=1, bias=self.bias)

        if pretrained == "download":
            if mode == "denoising":
                # denoising mode
                if self.denoising_mode == "real":
                    self.model_name = "real_denoising.pth"
                else:
                    if self.denoising_mode == "gray":
                        prefix_model_name = "gaussian_gray_denoising"
                    elif self.denoising_mode == "color":
                        prefix_model_name = "gaussian_color_denoising"
                    # sigma flavor
                    if self.denoising_sigma is None or self.denoising_sigma == "blind":
                        self.model_name = f"{prefix_model_name}_blind.pth"
                    if self.denoising_sigma == "15":
                        self.model_name = f"{prefix_model_name}_sigma15.pth"
                    if self.denoising_sigma == "25":
                        self.model_name = f"{prefix_model_name}_sigma25.pth"
                    if self.denoising_sigma == "50":
                        self.model_name = f"{prefix_model_name}_sigma50.pth"
                    else:
                        raise RuntimeError(
                            f"Inappropriate model sigma of '{self.denoising_sigma}'")
            elif mode == "deraining":
                self.model_name = "deraining.pth"
            elif mode == "defocus_deblurring":
                if self.dual_pixel_task:
                    self.model_name = "dual_pixel_defocus_deblurring.pth"
                else:
                    self.model_name = "single_image_defocus_deblurring.pth"

            url = get_weights_url(model_name="restormer",
                                  file_name=self.model_name)
            ckpt_restormer = torch.hub.load_state_dict_from_url(
                url, map_location=lambda storage, loc: storage, file_name=self.model_name
            )
            self.load_state_dict(ckpt_restormer, strict=True)

        elif os.path.exists(pretrained):
            ckpt_restormer = torch.load(
                pretrained, map_location=lambda storage, loc: storage
            )
            self.load_state_dict(ckpt_restormer, strict=True)

        elif pretrained is not None:
            raise ValueError(f"Inappropriate model name of '{pretrained}'")

        if not train:
            self.eval()
            for _, v in self.named_parameters():
                v.requires_grad = False

        if device == "gpu" and cuda:
            self.to(device)
        elif device == "gpu" and not cuda:
            raise RuntimeError("GPU mode requires cuda")

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1
