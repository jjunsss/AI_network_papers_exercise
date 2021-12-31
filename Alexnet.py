import torch
import torch.nn as nn
import torchvision 
from torchvision.transforms import transforms

from typing import Any
from tqdm import tqdm

BATCH_SIZE = 32
EPOCHES = 10

#* model pre-trained urls
model_urls = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
}

#* config model
class Alexnet(nn.Module):
    def __init__(self, num_classes : int = 1000) -> None:
        super().__init__()
        #torch._C._log_api_usage_once(self)  #! 어떤 이유로 사용하는지 모르겠음. config 관련
        
        self.layer = nn.Sequential(
            nn.Conv2d(3, 96, (11, 11), stride=4),    # 54 * 54  / 96 개의 channels 사용
            nn.ReLU(True),
            nn.MaxPool2d((3,3), stride=2),  # 26 * 26 * 96
            nn.Conv2d(96, 256, (5,5), padding=2),   # size가 증가함.
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), stride=2),
            nn.Conv2d(256, 384, (3,3), padding=1),
            nn.ReLU(True),
            nn.Conv2d(384, 384, (3,3), padding=1),
            nn.ReLU(True),
            nn.Conv2d(384, 256, (3,3), padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((3,3), 2)  #일부러 겹치게 풀링하는 것.
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(6400, 4096),  #Flatten 진행시에 값을 대략적으로 아무런 값을 넣고 진행하면 계산하지 않고도 답을 알 수 있다.
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)   #Final Dense
        )
        
    def forward(self, x) -> torch.Tensor:
        x = self.layer(x)
        x = torch.flatten(x, 1)    # Height * Width * channel value
        x = self.classifier(x)
        return x

model = Alexnet(10)

#pretrained. progressbar. classes (imageNet = 1000), 이거 내가 AlaxNet 수정해서 사용 x 
def alexnet(pretrained: bool = True, progress : bool = True, **kwargs: Any) -> Alexnet:
    model = Alexnet(**kwargs)
    
    if pretrained == True : 
        state_dict = torch.hub.load_state_dict_from_url(model_urls["alexnet"], progress=progress)
        model.load_state_dict(state_dict)
        return model
        

train_dataset = torchvision.datasets.CIFAR10(root = "../CIFAR_10", train=True, download=False, transform= transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))
test_dataset = torchvision.datasets.CIFAR10(root = "../CIFAR_10", train=False, download=False, transform= transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))

train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') #아마 label 선언이 없고 index로 나타내고 있음.

for (x, y) in test_dataloader:
    print(f'test image size {x.size()} \\\\\\\\\\\\\\\\t test label size {y.size()}')
    print(classes[y[0]])
    break


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train(model, train_dataloader, optimizer, criterion, epoch):
    model.train()
    running_loss: float = 0.0
    
    for idx, (images, labels) in enumerate(tqdm(train_dataloader, desc="training", position=0, leave= True)):
        optimizer.zero_grad()
        
        predict = model(images)
        loss = criterion(predict, labels)
        loss.backward()
        
        optimizer.step()
        
        
        running_loss += loss.item()
        if idx % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.4f' % (epoch + 1, idx + 1, running_loss / 2000))
            running_loss = 0.0
    
    
#def test():
    

for epoch in range(EPOCHES):
    train(model, train_dataloader, optimizer, criterion, epoch)
    
