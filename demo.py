import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

a = [[1, 2, 3, 4], [4, 5, 6, 7, 9], [6, 7, 8, 9, 4, 5], [4, 3, 2], [8, 7, 5, 4], [4, 8, 7, 1]]
b = [1, 2, 3, 4, 5, 6]


class demo(torch.nn.Module):
    def __init__(self, x, y):
        super(demo, self).__init__()
        self.x = x
        self.y = y
    def forward(self, input):
        pass



class mydataset(Dataset):
    def __init__(self, x, y):
        self.feature = x
        self.label = y

    def __getitem__(self, item):
        return torch.tensor(self.feature[item]), self.label[item]  # 根据需要进行设置

    def __len__(self):
        return len(self.feature)


dataset = mydataset(a, b)

for i in range(len(dataset)):
    print(dataset[i])


def fun(x):
    x.sort(key=lambda data:len(data[0]),reverse=True)
    print(x)
    feature = []
    label = []
    length = []
    for i in x:
        feature.append(i[0])
        label.append(i[1])
        length.append(len(i[0]))
    # feature = pad_sequence(feature,batch_first=True,padding_value=-1)     # 可以适当的进行补齐操作
    return torch.tensor(label), torch.tensor(length)


dataloader = DataLoader(dataset,batch_size=2,collate_fn=fun)    # 定义DataLoader

for y,length in dataloader:
    print(y, length)
    print(type(y), type(length))
    print('------------------')