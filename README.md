## Запуск
Основу программы составляют два файла: файл уже обученной нейросетевой модели model.pth и файл собственно программы prog.py. Программа подгружает модель и использует её для распознавания нарисованной пользователем цифры. 

На компьютере уже должны быть установлены python-библиотеки numpy, torch, pyglet. Об установке этих пакетов под конкретную систему смотрите официальную информацию в сети. В моём случае установка совершалась командой pip, например:
```
pip install pyglet
``` 
Если нужные библиотеки установлены, то для запуска программы разместите файлы model.pth и prog.py в один каталог, затем используйте команду:
```
python3 prog.py
```

![screen1](https://github.com/antgdnpr/mnist_digit_recognition/assets/154733297/7e7d5862-926c-4f37-9390-382f640fd1f1)

## Генерация файла модели
Файл model.pth уже готов для использования, но изначально для его создания была использована программа get_nn.py. Эта программа использует датасет MNIST для обучения полносвязной нейросети, после чего сохраняет натренированную модель в файле tmp_model.pth. 
Датасет MNIST доступен для скачивания по адресу http://yann.lecun.com/exdb/mnist/.



Команда
```
python3 get_nn.py
```
создаст файл tmp_model.pth, который можно использовать вместо model.pth. В коде программы-
