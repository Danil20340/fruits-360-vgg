# Fruits-360: классификация фруктов и овощей с помощью VGG16 (Transfer Learning)

Лабораторная работа № 10 по дисциплине «Нейронные сети».
Выполнил: Бронников Д.С., группа МОм-25-1.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Danil20340/fruits-360-vgg/blob/main/VGG16%20Fruits-360.ipynb)

Свёрточная часть VGG16 с весами ImageNet заморожена, поверх неё обучается новый классификатор
(GlobalAveragePooling → Dense 512 → Dropout 0,5 → Dense 131). Данные — датасет
[Fruits-360](https://github.com/Horea94/Fruit-Images-Dataset): 131 класс фруктов и овощей,
изображения 100×100 RGB, увеличенные до 224×224. Дополнительно в работе визуализированы активации
слоёв `block1_conv1` и `block5_conv3`, выполнен fine-tuning последнего свёрточного блока, а модель
сравнивается с CNN из [лабораторной работы № 9](https://github.com/Danil20340/fruits-360).

## Результаты

| Показатель | VGG16, Transfer Learning | + fine-tuning блока 5 |
|---|---|---|
| Accuracy на тестовой выборке | **98,39 %** (366 ошибок из 22 688) | **99,18 %** (187 ошибок) |
| Macro F1 | 0,984 | 0,992 |
| Время обучения на GPU T4 в Google Colab (mixed precision) | **1786 с (29,8 мин)**, 15 эпох | +657 с (11,0 мин), 5 эпох |
| Параметров, всего / обучаемых | 15,04 млн / 0,33 млн | 15,04 млн / 7,41 млн |

В репозитории сохранена модель с Transfer Learning (без fine-tuning), файл 62,5 МБ.

CNN из лабораторной работы № 9 на той же тестовой выборке даёт 99,17 % (189 ошибок) при обучении за 584 с.
С замороженной свёрточной частью VGG16 ей уступает, а fine-tuning последнего блока выводит VGG16 на тот же
уровень ценой в 4 раза большего времени обучения и в 8 раз более медленного инференса. Ошибки двух моделей
почти не пересекаются. Подробный анализ — в разделах 7.1, 8.1 и 12 ноутбука.

## Файлы

| Файл | Что это |
|---|---|
| [`VGG16 Fruits-360.ipynb`](VGG16%20Fruits-360.ipynb) | отчёт: подготовка данных, модель, обучение с замером времени, оценка качества, сохранение и загрузка модели, сравнение с CNN |
| `fruits360_vgg16_model.keras` | обученная модель VGG16 с Transfer Learning (Keras 3, формат `.keras`, 62,5 МБ) |
| `class_names_vgg.txt` | названия 131 класса в порядке выходов модели |

## Данные

Датасет в репозиторий не копируется: архив весит около 760 МБ, это больше лимита GitHub на один файл.
Ноутбук скачивает его по прямой ссылке
https://github.com/Horea94/Fruit-Images-Dataset/archive/refs/heads/master.zip

## Быстрый инференс

Предобработка VGG16 (`preprocess_input`) встроена в модель, поэтому на вход подаётся обычное
изображение RGB со значениями 0..255, увеличенное до 224×224 билинейной интерполяцией.

```python
import urllib.request

import numpy as np
import tensorflow as tf

base = "https://github.com/Danil20340/fruits-360-vgg/raw/main"
urllib.request.urlretrieve(f"{base}/fruits360_vgg16_model.keras", "fruits360_vgg16_model.keras")
urllib.request.urlretrieve(f"{base}/class_names_vgg.txt", "class_names_vgg.txt")

model = tf.keras.models.load_model("fruits360_vgg16_model.keras", compile=False)
with open("class_names_vgg.txt", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

img = tf.keras.utils.load_img("apple.jpg", target_size=(224, 224), interpolation="bilinear")
probs = model.predict(tf.expand_dims(tf.keras.utils.img_to_array(img), 0), verbose=0)[0]
print(class_names[np.argmax(probs)], f"{np.max(probs):.2%}")
```
