import random
import typing as tp
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchvision
from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import (
    _mobilenet_extractor,
    mobilenet_backbone,
)
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead
from torchvision.models.mobilenetv3 import (
    MobileNet_V3_Large_Weights,
    mobilenet_v3_large,
)
from torchvision.ops import misc
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_bounding_boxes

SupportedModels = tp.Literal[
    "fasterrcnn_mobilenet_v3_large_320_fpn",
    "fasterrcnn_mobilenet_v3_large_fpn",
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_resnet50_fpn_v2",
    "fcos_resnet50_fpn",
    "retinanet_resnet50_fpn",
    "retinanet_resnet50_fpn_v2",
    "retinanet_mobilenet_v3_large_fpn",
    "ssd300_vgg16",
    "ssdlite320_mobilenet_v3_large",
]
Devices = tp.Literal["cuda", "cpu"]
SAMtypes = tp.Literal["vit_h", "vit_l", "vit_b"]


def load_model(
    model_type: SupportedModels,
    num_classes: tp.Optional[int] = 2,
    device: Devices = "cuda",
    ckpt_path: tp.Optional[str] = None,
    pretrained: bool = False,
    pretrained_backbone: bool = False,
    anchor_sizes: tp.Optional[tp.Tuple[int]] = None,
    trainable_backbone_layers: tp.Optional[int] = None,
) -> nn.Module:
    """Load an object detection model. Note that this function restrains the
    combinations of model architectures and backbones to the default combinations given
    by pytorch. For instance, you can't load an SSD model with a Resnet50 backbone.

    Parameters
    ----------
    model_type : SupportedModels
    num_classes : tp.Optional[int], optional
        Including the background class. Overrided when the model is loaded with default
        weights. By default 2.
    device : Devices, optional
        Either cuda or cpu, by default "cuda".
    ckpt_path : tp.Optional[str], optional
        Path to the checkpoint, by default None.
    pretrained : bool, optional
        Whether to load the default weights for the given architecture. The list of
        default weights can be found at:
        https://pytorch.org/vision/stable/models.html#object-detection
        By default False.
    pretrained_backbone : bool, optional
        Whether to load the default pretrained backbone for the given architecture (e.g.
        a pretrained vgg16 backbone for the ssd300_vgg16 architecture. This parameter is
        ignored when pretrained = True. By default, False.
    anchor_sizes: tp.Optional[tp.Tuple[int]], optional
        Only used to instantiate a fasterrcnn_mobilenet_v3_large_fpn model. By default,
        None.
     trainable_backbone_layers: tp.Optional[int], optional
        Only used to instantiate a fasterrcnn_mobilenet_v3_large_fpn model. By default,
        None.

    Returns
    -------
        Object detector model
    """

    # Instantiate model
    if model_type == "retinanet_mobilenet_v3_large_fpn":  # custom model
        model = load_retinanet_mobilenet_v3_large(
            num_classes, pretrained_backbone, True
        )
    elif model_type == "fasterrcnn_mobilenet_v3_large_fpn":  # custom model
        model = load_fasterrcnn_mobilenet_v3_large_fpn(
            num_classes,
            pretrained_backbone,
            anchor_sizes=anchor_sizes,
            trainable_backbone_layers=trainable_backbone_layers,
        )
    else:
        model = torchvision.models.get_model(
            model_type,
            pretrained=pretrained,
            pretrained_backbone=pretrained_backbone,
            num_classes=num_classes,
        )

    # Load weights
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])

    # Device
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    print(f"{model_type} loaded to {device}")

    return model


def load_fasterrcnn_mobilenet_v3_large_fpn(
    num_classes: int,
    pretrained_backbone: bool,
    anchor_sizes: tp.Tuple[int] = (32, 64, 128, 256, 512),
    trainable_backbone_layers: int = 3,
    **kwargs: tp.Any,
) -> FasterRCNN:
    """Loads a faster RCNN with mobilenet_v3_large_fpn backbone with custom settings.

    Parameters
    ----------
    num_classes : int
    pretrained_backbone : bool
    anchor_sizes : tp.Tuple[int]
        Default value in torchvision is (32, 64, 128, 256, 512). The size is given as
        the side length of a square box, in the input space (i.e. pixels).
    trainable_backbone_layers : int, optional
        MobileNet v3 large has 6 layers. When pretrained_backbone is False, this
        parameter is ignored and all layers are trained. The default value is the one
        found in torchvision: the first 3 layers are frozen by default when loading the
        pretrained backbone.
    """

    if pretrained_backbone:
        weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V1
    else:
        weights_backbone = None
        trainable_backbone_layers = 6  # all layers are trained

    norm_layer = misc.FrozenBatchNorm2d if pretrained_backbone else nn.BatchNorm2d
    backbone = mobilenet_v3_large(weights=weights_backbone, norm_layer=norm_layer)
    backbone = _mobilenet_extractor(backbone, True, trainable_backbone_layers)
    anchor_sizes = (anchor_sizes,) * 3
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    kwargs = {"rpn_score_thresh": 0.05, **kwargs}
    model = FasterRCNN(
        backbone,
        num_classes,
        rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
        **kwargs,
    )

    return model


def load_retinanet_mobilenet_v3_large(
    num_classes: int, pretrained_backbone: bool, fpn: bool
):
    backbone = mobilenet_backbone(
        "mobilenet_v3_large",
        weights="IMAGENET1K_V2" if pretrained_backbone else None,
        fpn=fpn,
    )
    anchor_generator = _default_anchorgen()
    head = RetinaNetHead(
        backbone.out_channels,
        anchor_generator.num_anchors_per_location()[0],
        num_classes,
        norm_layer=partial(nn.GroupNorm, 32),
    )
    head.regression_head._loss_type = "giou"
    model = RetinaNet(
        backbone, num_classes, anchor_generator=anchor_generator, head=head
    )
    return model


def _default_anchorgen():
    anchor_sizes = tuple(
        (x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128]
    )
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    return anchor_generator


def compute_model_size(model: nn.Module) -> float:
    """Returns model size in megabytes"""

    mem_params = (
        sum([param.nelement() * param.element_size() for param in model.parameters()])
        / 10**6
    )
    mem_bufs = (
        sum([buf.nelement() * buf.element_size() for buf in model.buffers()]) / 10**6
    )

    return mem_params + mem_bufs


def set_random_seed(random_seed):
    """Set random seed for reproducibility"""
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        # This possibly comes at the cost of performance
        torch.backends.cudnn.benchmark = False


def show(sample):
    """Little function to display a sample from a dataset (one with bounding boxes, e.g.
    TrainingDataset and SplitDataset). Mostly for sanity checks.
    """
    image, target = sample
    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)
    image = F.convert_dtype(image, torch.uint8)
    annotated_image = draw_bounding_boxes(
        image, target["boxes"], colors="yellow", width=3
    )
    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()

    fig.show()


def bounding_box_to_tensor(batch: tp.List[tp.Dict[str, tp.Any]]):
    """Pytorch can be stupid. The torchvision.transforms.v2 operates on BoundingBox
    objects (a torch.Tensor subclass). On the other hand, torchmetrics.detection.mean_ap
    needs tensors as input. Hence the need to convert bounding boxes from BoundingBox to
    Tensor.

    Parameters
    ----------
    batch : list[dict]
        As returned by the dataloader when iterating. Each dict has two fields:
        - 'boxes' (expected to be a BoundingBox)
        - 'labels' (Int64Tensor)
    """
    converted_batch = []
    for preds in batch:
        converted_preds = {
            "boxes": preds["boxes"].as_subclass(torch.Tensor),
            "labels": preds["labels"],
        }
        converted_batch.append(converted_preds)
    return converted_batch


def preds_to_ground_truth(preds: tp.Dict[str, tp.List], score_thr: float):
    """Converts predictions to groud truths by applying a threshold on the score (below
    the threshold, bboxes are considered negatives and discarted, while above the score,
    they are considered positives and are kept). Images with no ground truth data are
    filtered out.

    Parameters
    ----------
    preds : dict[str, list]
        The keys of the dict are the image IDs. For each image, a list with 3 elements
        is expected, in the following order:
        - a list of bounding boxes in the [x1, y1, x2, y2] format, with 0 <= x1 < x2 <=
        W and 0 <= y1 < y2 <= H
        - a list of labels
        - a list with the corresponding scores

    score_thr : float

    Returns
    -------
    ground_truths : dict[str, list]
        Same as the 'preds' parameter, but withouth the scores, and only including boxes
        (and corresponding labels) with score above the threshold.
    """

    ground_truths = {}
    for img_id, img_preds in preds.items():
        bboxes = np.array(img_preds[0])
        labels = np.array(img_preds[1])
        scores = np.array(img_preds[2])

        bboxes = bboxes[scores >= score_thr]
        labels = labels[scores >= score_thr]

        if len(labels) != 0:
            ground_truths[img_id] = [bboxes.tolist(), labels.tolist()]

    nb_images_filtered_out = len(preds) - len(ground_truths)
    if nb_images_filtered_out > 0:
        print(
            f"Warning: {nb_images_filtered_out} images filtered out (no bounding boxes)"
        )

    return ground_truths
