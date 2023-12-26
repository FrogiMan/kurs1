import telebot
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

# Пути к данным
data_dir = 'fridges'
model_path = 'fridge_torch.pth'

# Загрузка обученной модели
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 38)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Преобразования изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Инициализация токена бота
bot_token = '6730708583:AAEPPGr5nqsvtabaH-B_RVf2ASGTHhPKxd8'
bot = telebot.TeleBot(bot_token)
# Обработчик команды /start
@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.reply_to(message, "Hello! I am your image classification bot. Send me a photo, and I will tell you the top classes.")

# Обработчик команды /help
@bot.message_handler(commands=['help'])
def handle_help(message):
    bot.reply_to(message, "Send me a photo, and I will provide you with the top classes predicted by the model.")

# Обработчик изображений
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    # Получение файла из сообщения пользователя
    file_info = bot.get_file(message.photo[-1].file_id)
    image_stream = io.BytesIO(bot.download_file(file_info.file_path))

    # Преобразование изображения
    image = Image.open(image_stream)
    image_tensor = transform(image).unsqueeze(0)

    # Получение предсказания
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Сопоставление индексов классов с названиями
    class_names = [
        'alcohol', 'apple', 'banana', 'beetroot', 'bell pepper', 'bread', 'cabbage', 'carrot', 'cauliflower',
        'cheese', 'chilli pepper', 'cucumber', 'eggplant', 'eggs', 'garlic', 'ginger', 'grapes', 'juice', 'kiwi',
        'lemon', 'lettuce', 'mango', 'meat', 'milk', 'orange', 'other milk products', 'pear', 'peas', 'pineapple',
        'pomegranate', 'raddish', 'sauce', 'sausage', 'sparkling water', 'spinach', 'tomato', 'water', 'watermelon'
    ]

    # Получение топ-N классов
    top_n = 3
    top_classes = torch.argsort(probabilities, descending=True)[:top_n]

    # Формирование сообщения с результатами
    result_message = "Top classes:\n"
    for i, class_index in enumerate(top_classes):
        class_name = class_names[class_index]
        probability_percentage = round(float(probabilities[class_index]) * 100, 2)
        result_message += f"{i + 1}. {class_name}: {probability_percentage}%\n"

    # Отправка результата пользователю
    bot.reply_to(message, result_message)

if __name__ == '__main__':
    bot.polling(none_stop=True)