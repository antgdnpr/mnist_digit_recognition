import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

where_data = '...' # ВСТАВЬТЕ СЮДА ПУТЬ К КАТАЛОГУ С ОБРАБОТАННЫМИ БИНАРНЫМИ ФАЙЛАМИ ДАТАСЕТА (СМ. README)

images = torch.from_file(where_data+'images', size=70000*28*28, shared=False, dtype=torch.uint8) / 255
labels = torch.from_file(where_data+'labels', size=70000, shared=False, dtype=torch.uint8)
images = images.reshape(-1, 28, 28)

test_size = 0.2
test_border = int(labels.shape[0] * (1.0 - test_size))


# Класс для подвыборок train и test.
class CustomDataset(Dataset):

    def __init__(self, tmp_mode):
        if tmp_mode == 'train':
            self.images = images[:test_border, :, :]
            self.labels = labels[:test_border]
        if tmp_mode == 'test':
            self.images = images[test_border:, :, :]
            self.labels = labels[test_border:]
            
    def __len__(self):
        return self.labels.shape[0]
        
    def __getitem__(self, idx):
        return self.images[idx, :, :], self.labels[idx]


# Описание архитектуры модели.
class MyNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Dropout(0.4),
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


# Функции обучения и тестирования.
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0    
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f'(test) \t accuracy = {(100*correct):4.1f}% \t average loss = {test_loss:8.4f}')
        
        
model = MyNN() 

# Гиперпараметры.
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.05
batch_size = 100

# Конструирование объектов подвыборки для train и test.
train_data = CustomDataset('train')
test_data = CustomDataset('test')
train_dl = DataLoader(train_data, batch_size=batch_size)
test_dl = DataLoader(test_data, batch_size=batch_size)

# Обучение модели.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
epochs = 30
for t in range(epochs):
    print(f'Epoch {t+1:2d}: \t', end='')
    train_loop(train_dl, model, loss_fn, optimizer)
    test_loop(test_dl, model, loss_fn)

# Сохранение модели.
torch.save(model, 'tmp_model.pth')    
