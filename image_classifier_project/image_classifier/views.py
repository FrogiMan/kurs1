import time
from django.shortcuts import render
from .forms import UploadedImageForm
import torch
from torchvision import transforms, models
from PIL import Image

# Пути к данным
model_path = 'C:/Users/safro/PycharmProjects/fridge_torch/image_classifier_project/fridge_torch.pth'
class_names = [
    'alcohol', 'apple', 'banana', 'beetroot', 'bell pepper', 'bread', 'cabbage', 'carrot', 'cauliflower',
    'cheese', 'chilli pepper', 'cucumber', 'eggplant', 'eggs', 'garlic', 'ginger', 'grapes', 'juice', 'kiwi',
    'lemon', 'lettuce', 'mango', 'meat', 'milk', 'orange', 'other milk products', 'pear', 'peas', 'pineapple',
    'pomegranate', 'raddish', 'sauce', 'sausage', 'sparkling water', 'spinach', 'tomato', 'water', 'watermelon'
]


def predict_resnet18(image_path, top_n=3):
    # Загрузка предварительно обученной модели ResNet18
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 38)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Преобразование изображения
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    # Получение предсказания
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_classes = torch.argsort(probabilities, descending=True)[:top_n]

        # Сопоставление индексов классов с их названиями
        predicted_classes = [class_names[class_index] for class_index in top_classes]
        probabilities = [round(float(probabilities[class_index]) * 100, 2) for class_index in top_classes]

        return list(zip(predicted_classes, probabilities))


def upload_image(request):
    image_url = None
    results = None

    if request.method == 'POST' and request.FILES['image']:
        form = UploadedImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.save()

            # Опционально, можно добавить timestamp к URL, чтобы избежать кэширования изображения
            image_url = image.image.url + f'?timestamp={int(time.time())}'

            # Получите предсказание
            results = predict_resnet18(image.image.path, top_n=3)

        return render(request, 'myapp/upload_image.html', {'image_url': image_url, 'results': results})
    else:
        form = UploadedImageForm()
        return render(request, 'myapp/upload_image.html', {'image_url': image_url, 'form': form})