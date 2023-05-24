import torch
from cnn_resnet import ResNetCNN, load_model, predict, labels
from torchvision import transforms
from PIL import Image

model_path = "my_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetCNN(10).to(device)
load_model(model, model_path)


def _preprocess_image(image_path, image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if image is None:
        image = Image.open(image_path)
    image = transform(image)
    return image.unsqueeze(0)


def predict_image(image_path=None, image=None):
    image = _preprocess_image(image_path=image_path, image=image)
    prediction = predict(model, image, device)
    prediction = labels[prediction]
    return prediction
