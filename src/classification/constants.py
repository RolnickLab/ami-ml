#!/usr/bin/env python
# coding: utf-8

"""Constants accessible in the entire project
"""

EFFICIENTNETV2_B3 = "tf_efficientnetv2_b3"
EFFICIENTNETV2_S_IN21K = "tf_efficientnetv2_s_in21k"
SWIN_S = "swin_small_patch4_window7_224"
MOBILENETV3LARGE = "mobilenetv3_large_100"
RESNET50 = "resnet50"
CONVNEXT_T = "convnext_tiny_in22k"
CONVNEXT_B = "convnext_base_in22k"
VIT_B16_128 = "vit_base_patch16_128_in21k"
VIT_B16_224 = "vit_base_patch16_224_in21k"
VIT_B16_384 = "vit_base_patch16_384"

ADAMW = "adamw"
SGD = "sgd"

CROSS_ENTROPY_LOSS = "cross_entropy"

COSINE_LR_SCHEDULER = "cosine"

AVAILABLE_MODELS = frozenset(
    [
        EFFICIENTNETV2_B3,
        EFFICIENTNETV2_S_IN21K,
        SWIN_S,
        MOBILENETV3LARGE,
        RESNET50,
        CONVNEXT_T,
        CONVNEXT_B,
        VIT_B16_128,
        VIT_B16_224,
        VIT_B16_384,
    ]
)


AVAILABLE_LOSS_FUNCIONS = frozenset([CROSS_ENTROPY_LOSS])

AVAILABLE_OPTIMIZERS = frozenset([ADAMW, SGD])

AVAILABLE_LR_SCHEDULERS = frozenset([COSINE_LR_SCHEDULER])
