import pyglet
import torch
import numpy as np
from time import sleep

wd = pyglet.window.Window(640, 720)

ar = np.zeros((1, 28, 28), dtype='u1')
exitflag, beginflag = 0, 0


# Установка "флага выхода" при попытке закрыть графическое окно.
@wd.event
def on_close():
    global exitflag
    exitflag = 1


# Очистка окна при нажатии правой кнопки мыши.
@wd.event
def on_mouse_press(x, y, button, modifiers):
    global ar, beginflag
    if button == pyglet.window.mouse.RIGHT:
        ar = np.zeros((1, 28, 28), dtype='u1')
        pyglet.shapes.Rectangle(40, 120, 560, 560, color=(0, 0, 0)).draw()
        beginflag = 0
        erase_label().draw()
        gui_label(f'Начните рисовать цифру в окне', 90, 200).draw()


# Рисование левой кнопкой мыши.
@wd.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    diap = list(range(28))
    global ar, beginflag
    if buttons == pyglet.window.mouse.LEFT:
        xq = (x -  40) // 20    # Интересует xq и yq из 0..27 - это номера "больших" пикселей
        yq = (y - 120) // 20
        for (xq, yq) in [(xq, yq), (xq-1, yq), (xq+1, yq), (xq, yq-1), (xq, yq+1)]:
            if (xq in diap) and (yq in diap):
                beginflag = 1
                ar[0, 28-yq-1, xq] = 255
                pyglet.shapes.Rectangle(40 + 20*xq, 120 + 20*yq, 20, 20, color=(200, 200, 200)).draw()


# Создание текстового блока.
def gui_label(s, y, brightness):
    return pyglet.text.Label(s,
                            font_name='Monospace Regular',
                            font_size=14,
                            x=40, y=y,
                            color=(brightness, brightness, brightness, 255))


# Перекрытие текстового блока для его последующей перезаписи.                            
def erase_label():
    return pyglet.shapes.Rectangle(5, 85, 708, 30, color=(0, 0, 0))


# Описание архитектуры той модели, которая будет подгружена.
class MyNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
        
        
# Оставшийся код - начальная графика, загрузка модели и цикл опроса GUI.

pyglet.shapes.Box(39, 119, 562, 562, color=(200, 200, 200), thickness=1).draw()
gui_label(f'Начните рисовать цифру в окне', 90, 200).draw()
gui_label(f'Для очистки окна нажмите правую кнопку мыши', 65, 140).draw()
wd.set_visible()

model = torch.load('model.pth')
model.eval()

with torch.no_grad():
    while True:
        sleep(0.025)
        wd.dispatch_events()
        wd.flip()
        if exitflag:
            break
        if not beginflag:
            continue
        image = torch.from_numpy(ar).to(dtype=torch.float)
        raw_res = model(image)
        ar_res = (torch.nn.Softmax(dim=1)(raw_res)).numpy().flatten()
        res = np.argmax(ar_res)
        erase_label().draw()
        gui_label(f'Распознано как цифра <{res}>', 90, 200).draw()
        

