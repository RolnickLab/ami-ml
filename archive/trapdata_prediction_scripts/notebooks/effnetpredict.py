import json

from PIL import Image
from torchvision import transforms
import timm
import torch

class ModelInference:
  def __init__(self, model_path, category_map_json, device, input_size=300):
    self.device = device
    self.input_size = input_size
    self.id2categ = self._load_category_map(category_map_json)
    self.transforms = self._get_transforms()
    self.model = self._load_model(model_path, num_classes=len(self.id2categ))
    self.model.eval()

  def _load_category_map(self, category_map_json):
    with open(category_map_json, 'r') as f:
      categories_map = json.load(f)

    id2categ = {categories_map[categ]: categ for categ in categories_map}

    return id2categ

  def _get_transforms(self):
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    return transforms.Compose([
      transforms.Resize((self.input_size, self.input_size)),
      transforms.ToTensor(),
      transforms.Normalize(mean, std),
      ])

  def _load_model(self, model_path, num_classes):
    model = timm.create_model('tf_efficientnetv2_b3',
                              pretrained=False,
                              num_classes=num_classes)
    model = model.to(self.device)
    model.load_state_dict(torch.load(model_path,
                                     map_location=torch.device(self.device)))

    return model

  def predict(self, image, confidence=False):
    with torch.no_grad():
      image = self.transforms(image)
      image = image.to(self.device) 
      image = image.unsqueeze_(0)

      predictions = self.model(image)
      predictions = torch.nn.functional.softmax(predictions, dim=1)
      predictions = predictions.cpu().numpy()

      categ = predictions.argmax(axis=1)[0]
      categ = self.id2categ[categ]

      if confidence:
        return categ, predictions.max(axis=1)[0]
      else:
        return categ


def main():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(device)

  image_path = '/network/scratch/a/aditya.jain/GBIF_Data/moths_quebec-vermont/Noctuidae/Amphipoea/Amphipoea americana/2251271946.jpg'
  category_map_json = '/home/mila/f/fagner.cunha/repositories/lepsAI/classification/data/03-mothsv2_category_map.json'
  model_path = '/home/mila/f/fagner.cunha/scratch/training/ckp-mothsv2/model_20220421_110638_30.pth'

  image = Image.open(image_path)

  model_inf = ModelInference(model_path, category_map_json, device)

  categ = model_inf.predict(image)
  print(f'Prediction: {categ}')

  categ, conf = model_inf.predict(image, confidence=True)
  print(f'Prediction: {categ}, Confidence: {conf}')


if __name__ == '__main__':
  main()
