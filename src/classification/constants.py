#!/usr/bin/env python
# coding: utf-8

"""Constants to be used within the project
"""

EFFICIENTNETV2_B3 = "efficientnetv2_b3"
EFFICIENTNETV2_S_IN21K = "efficientnetv2_s_in21k"
SWIN_S = "swin_small_patch4_window7_224"
RESNET50 = "resnet50"
TIMM_MOBILENETV3LARGE = "mobilenetv3_large_100"
TIMM_RESNET50 = "resnet50"
TIMM_CONVNEXT_T = "convnext_tiny_in22k"
TIMM_CONVNEXT_B = "convnext_base_in22k"
TIMM_VIT_B16_128 = "vit_base_patch16_224_in21k"
TIMM_VIT_B16_224 = "vit_base_patch16_224_in21k"
TIMM_VIT_B16_384 = "vit_base_patch16_224_in21k"

AVAILABLE_MODELS = frozenset(
    [
        EFFICIENTNETV2_B3,
        EFFICIENTNETV2_S_IN21K,
        SWIN_S,
        RESNET50,
        TIMM_MOBILENETV3LARGE,
        TIMM_RESNET50,
        TIMM_CONVNEXT_T,
        TIMM_CONVNEXT_B,
        TIMM_VIT_B16_128,
        TIMM_VIT_B16_224,
        TIMM_VIT_B16_384,
    ]
)
