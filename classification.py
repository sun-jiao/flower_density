from io import BytesIO

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


label_list = ['flower1', 'flower2']

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = None


def get_model():
    _model = models.resnet34(weights=None)
    num_ftrs = _model.classifier[1].in_features
    _model.classifier[1] = torch.nn.Linear(num_ftrs, len(label_list))
    _model.load_state_dict(torch.load('resnet34.pth', map_location=torch.device('cpu')))
    _model.eval()

    return _model


def predict(file):
    global model
    if model is None:
        model = get_model()

    image = Image.open(BytesIO(file.read())).convert('RGB')
    image = transform(image)
    image = torch.unsqueeze(image, 0)
    outputs = model(image)

    # get top 3 results and their probability
    probs, indices = torch.topk(outputs, k=len(label_list), dim=1)
    probs = torch.nn.functional.softmax(probs, 1)
    probs = probs.squeeze().tolist()
    indices = indices.squeeze().tolist()

    result_list = []

    for index in range(3):
        result_list.append({
            'name': label_list[indices[index]],
            'prob': round(probs[index] / sum(probs), 4)
        })

    return result_list
