import torch.nn as nn
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


class LoRAImageEncoderAdapter(nn.Module):
    def __init__(self, ori_sam: Sam, fix=False, lora_r=4, lora_alpha=16):
        super().__init__()
        # load the SAM image encoder
        # freeze original weights
        self.sam_img_encoder = ori_sam.image_encoder

        for p in self.sam_img_encoder.parameters():
            p.requires_grad = False
        # configure LoRA for all Transformer self-attn
        peft_config = LoraConfig(
            task_type=TaskType.VISION_ENCODER,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],  # inject into query & value projections
        )
        # wrap the image encoder
        self.sam_img_encoder = get_peft_model(self.sam_img_encoder, peft_config)

    def forward(self, x):
        return self.sam_img_encoder(x)

