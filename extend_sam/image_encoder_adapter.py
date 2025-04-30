import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

import math

from .segment_anything_ori.modeling.sam import Sam
from peft import LoraConfig, get_peft_model, TaskType
from .utils import fix_params


class BaseImgEncodeAdapter(nn.Module):

    def __init__(self, ori_sam: Sam, fix=False):
        super(BaseImgEncodeAdapter, self).__init__()
        self.sam_img_encoder = ori_sam.image_encoder
        if fix:
            fix_params(self.sam_img_encoder)

    def forward(self, x):
        x = self.sam_img_encoder(x)
        return x

class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv

def fix_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

class LoRAImageEncoderAdapter(nn.Module):
    def __init__(self, ori_sam: Sam, fix=False, lora_r=4, lora_alpha=16, lora_layer=None):
        super().__init__()
        # load the SAM image encoder
        self.sam_img_encoder = ori_sam.image_encoder

        if fix:
            fix_params(self.sam_img_encoder)

        for p in self.sam_img_encoder.parameters():
            p.requires_grad = False

        # Set the layers to apply LoRA to
        self.lora_layer = lora_layer if lora_layer is not None else list(range(len(self.sam_img_encoder.blocks)))
        self.w_As = []
        self.w_Bs = []

        # configure LoRA for Transformer self-attn blocks
        for t_layer_i, blk in enumerate(self.sam_img_encoder.blocks):
            # If we only want few lora layers instead of all, skip others
            if t_layer_i not in self.lora_layer:
                continue
            print(f"Injecting LoRA into layer {t_layer_i} (block {blk})")

            # Apply LoRA on the qkv projection layer
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, lora_r, bias=False)
            w_b_linear_q = nn.Linear(lora_r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, lora_r, bias=False)
            w_b_linear_v = nn.Linear(lora_r, self.dim, bias=False)

            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters of LoRA layers."""
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x):
        x = self.sam_img_encoder(x)
        return x

# class LoRAImageEncoderAdapter(nn.Module):
#     def __init__(self, ori_sam: Sam, fix=False, lora_r=4, lora_alpha=16):
#         super().__init__()
#         # load the SAM image encoder
#         # freeze original weights
#         self.sam_img_encoder = ori_sam.image_encoder

#         for name, module in self.sam_img_encoder.named_modules():
#             print(name)

#         if fix:
#             fix_params(self.sam_img_encoder)

#         for p in self.sam_img_encoder.parameters():
#             p.requires_grad = False
#         # configure LoRA for all Transformer self-attn
#         peft_config = LoraConfig(
#             task_type=TaskType.FEATURE_EXTRACTION,
#             inference_mode=False,
#             r=lora_r,
#             lora_alpha=lora_alpha,
#             target_modules=["qkv"],  # inject into query & value projections
#         )
#         # wrap the image encoder
#         self.sam_img_encoder = get_peft_model(self.sam_img_encoder, peft_config)

#     def forward(self, x):
#         return self.sam_img_encoder(x)

