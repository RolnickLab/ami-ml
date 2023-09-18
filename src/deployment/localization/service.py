import logging
import os
import pathlib
import tempfile
import urllib.request

import bentoml
import bentoml.io
import PIL.Image
import torch
import torchvision
import torchvision.datapoints
import torchvision.models.detection.anchor_utils
import torchvision.models.detection.backbone_utils
import torchvision.models.detection.faster_rcnn
import torchvision.models.mobilenetv3
import torchvision.transforms.v2 as T
from fastapi import FastAPI
from pydantic import BaseModel

tempdir = tempfile.TemporaryDirectory()
os.environ["LOCAL_WEIGHTS_PATH"] = "./tmp/images"

CHECKPOINT = "https://object-arbutus.cloud.computecanada.ca/ami-models/moths/localization/fasterrcnn_mobilenet_v3_large_fpn_uqfh7u9w.pt"
SCORE_THRESHOLD = 0.5

logger = logging.getLogger(__name__)
BaseModel.model_config["protected_namespaces"] = ()


class LabelStudioTask(BaseModel):
    data: dict
    id: int


# Label studio task result
example_result = (
    {
        "from_name": "detected_object",
        "to_name": "image",
        "type": "rectangle",
        "value": {
            "x": 66.12516045570374,
            "y": 74.68075222439236,
            "width": 1.3237595558166504,
            "height": 2.0854356553819446,
        },
        "score": 0.8321816325187683,
    },
)


def get_or_download_file(path, destination_dir=None, prefix=None) -> pathlib.Path:
    """
    >>> filename, headers = get_weights("https://drive.google.com/file/d/1KdQc56WtnMWX9PUapy6cS0CdjC8VSdVe/view?usp=sharing")

    Taken from https://github.com/RolnickLab/ami-data-companion/blob/main/trapdata/ml/utils.py
    """
    if not path:
        raise Exception("Specify a URL or path to fetch file from.")

    # If path is a local path instead of a URL then urlretrieve will just return that path
    destination_dir = destination_dir or os.environ.get("LOCAL_WEIGHTS_PATH")
    fname = path.rsplit("/", 1)[-1]
    if destination_dir:
        destination_dir = pathlib.Path(destination_dir)
        if prefix:
            destination_dir = destination_dir / prefix
        if not destination_dir.exists():
            logger.info(f"Creating local directory {str(destination_dir)}")
            destination_dir.mkdir(parents=True, exist_ok=True)
        local_filepath = pathlib.Path(destination_dir) / fname
    else:
        raise Exception(
            "No destination directory specified by LOCAL_WEIGHTS_PATH or app settings."
        )

    if local_filepath and local_filepath.exists():
        logger.info(f"Using existing {local_filepath}")
        return local_filepath

    else:
        logger.info(f"Downloading {path} to {destination_dir}")
        resulting_filepath, headers = urllib.request.urlretrieve(
            url=path, filename=local_filepath
        )
        resulting_filepath = pathlib.Path(resulting_filepath)
        logger.info(f"Downloaded to {resulting_filepath}")
        return resulting_filepath


class LabelStudioRequest(BaseModel):
    tasks: list[LabelStudioTask]
    model_version: str
    project: str
    label_config: str
    params: dict


def load_model_scratch(
    checkpoint_path: str | pathlib.Path,
    trainable_backbone_layers: int = 6,
    anchor_sizes: tuple = (64, 128, 256, 512),
    num_classes: int = 2,
    device: str | torch.device = "cpu",
):
    norm_layer = torch.nn.BatchNorm2d
    backbone = torchvision.models.mobilenetv3.mobilenet_v3_large(
        weights=None, norm_layer=norm_layer
    )
    backbone = torchvision.models.detection.backbone_utils._mobilenet_extractor(
        backbone, True, trainable_backbone_layers
    )
    anchor_sizes = (anchor_sizes,) * 3
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    model = torchvision.models.detection.faster_rcnn.FasterRCNN(
        backbone,
        num_classes,
        rpn_anchor_generator=torchvision.models.detection.anchor_utils.AnchorGenerator(
            anchor_sizes, aspect_ratios
        ),
        rpn_score_thresh=0.05,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict") or checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def post_process_single(output: dict) -> tuple[list, list, list]:
    scores = output["scores"].cpu().detach().numpy().tolist()
    labels = output["labels"].cpu().detach().numpy().tolist()

    # This model does not use the labels from the object detection model
    assert all([label == 1 for label in labels])

    # Filter out objects if their score is under score threshold
    bboxes = output["boxes"][output["scores"] > SCORE_THRESHOLD]

    print(
        f"Keeping {len(bboxes)} out of {len(output['boxes'])} objects found (threshold: {SCORE_THRESHOLD})"
    )

    bboxes = bboxes.cpu().detach().numpy().tolist()
    return bboxes, labels, scores


def format_predictions_single(
    image: torchvision.datapoints.Image, bboxes, scores
) -> list[dict]:
    width, height = image.spatial_size
    return [
        {
            "from_name": "detected_object",
            "to_name": "image",
            "type": "rectangle",
            "value": {
                "x": bbox[0] / width * 100,
                "y": bbox[1] / height * 100,
                "width": (bbox[2] - bbox[0]) / width * 100,
                "height": (bbox[3] - bbox[1]) / height * 100,
            },
            "score": score,
            "original_width": width,
            "original_height": height,
            "image_rotation": 0,
        }
        for bbox, score in zip(bboxes, scores)
    ]


class Yolov5Runnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.model = load_model_scratch(
            checkpoint_path=get_or_download_file(CHECKPOINT),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def transform(self) -> T.Compose:
        return T.Compose([T.ToImageTensor(), T.ConvertImageDtype()])

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    @torch.no_grad()
    def inference(self, input_img_paths):
        input_imgs = [
            get_or_download_file(path) for path in input_img_paths if path is not None
        ]
        input_imgs = [PIL.Image.open(path) for path in input_imgs]
        input_imgs = self.transform()(input_imgs)
        results = self.model(input_imgs)
        results = [post_process_single(result) for result in results]
        predictions = [
            format_predictions_single(image, bboxes, scores)
            for image, (bboxes, _, scores) in zip(input_imgs, results)
        ]
        return predictions

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def render(self, input_imgs):
        # Return images with boxes and labels
        # convert image fom JpegImageFile to uint8  tensor to avoid error in draw_bounding_boxes
        to_tensor = T.Compose([T.ToImageTensor(), T.ToDtype(torch.uint8)])
        input_imgs_t = self.transform()(input_imgs)
        results = self.model(input_imgs_t)
        draw = torchvision.utils.draw_bounding_boxes
        to_image = torchvision.transforms.ToPILImage()
        out_imgs = [
            draw(to_tensor(image), output["boxes"])
            for image, output in zip(input_imgs, results)
        ]
        # overlay bounding boxes on original image
        out_imgs = [
            to_image(img * 0.5 + image * 0.5)
            for img, image in zip(out_imgs, input_imgs_t)
        ]

        return out_imgs


yolo_v5_runner = bentoml.Runner(Yolov5Runnable, max_batch_size=30)

svc = bentoml.Service("yolo_v5_demo", runners=[yolo_v5_runner])


@svc.on_startup
def download(_: bentoml.Context):
    get_or_download_file(CHECKPOINT)


@svc.api(input=bentoml.io.Image(), output=bentoml.io.JSON())
async def invocation(input_img):
    batch_ret = await yolo_v5_runner.inference.async_run([input_img])
    return batch_ret[0]


@svc.api(input=bentoml.io.Image(), output=bentoml.io.Image())
async def render(input_img):
    batch_ret = await yolo_v5_runner.render.async_run([input_img])
    return batch_ret[0]


input_spec = bentoml.io.JSON(pydantic_model=LabelStudioRequest)
output_spec = bentoml.io.JSON()


@svc.api(
    input=input_spec,
    output=output_spec,
)
async def predict(input_data):
    tasks = input_data.tasks
    image_paths = [task.data["image"] for task in tasks]
    task_predictions = await yolo_v5_runner.inference.async_run(image_paths)
    resp = {
        "results": [{"result": task_prediction} for task_prediction in task_predictions]
    }
    print(resp)
    return resp


fastapi_app = FastAPI()
svc.mount_asgi_app(fastapi_app)


# Health check endpoint to match what Label Studio expects
@fastapi_app.get("/health")
def health():
    return {
        "status": "UP",
        "model_class": "HighwayMoths",
    }


# Setup endpoint to match what Label Studio expects
# https://github.com/HumanSignal/label-studio-ml-backend/blob/master/label_studio_ml/api.py#L65
@fastapi_app.post("/setup")
def setup():
    return {
        "model_version": "2.0",
    }
