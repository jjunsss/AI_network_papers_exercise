import enum
from xmlrpc.client import Boolean, boolean
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

from torchvision.transforms import transforms
from torchsummary import summary

from typing import Any
from tqdm import tqdm

BATCH_SIZE = 32
EPOCHES = 10

#! config model
class Alexnet(nn.Module):
    def __init__(self, num_classes : int = 1000, BN_effect : Boolean = True) -> None:
        super().__init__()
        #torch._C._log_api_usage_once(self)  #! 어떤 이유로 사용하는지 모르겠음. config 관련
        if BN_effect == True :
            self.layer = nn.Sequential(
                nn.Conv2d(3, 96, (11, 11), stride=4),    # 54 * 54  / 96 개의 channels 사용
                nn.BatchNorm2d(96),
                nn.ReLU(True),
                nn.MaxPool2d((3,3), stride=2),  # 26 * 26 * 96
                
                nn.Conv2d(96, 256, (5,5), padding=2),   # size가 증가함.
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.MaxPool2d((3, 3), stride=2),
                
                nn.Conv2d(256, 384, (3,3), padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU(True),
                
                nn.Conv2d(384, 384, (3,3), padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU(True),
                
                nn.Conv2d(384, 256, (3,3), padding=1),
                nn.BatchNorm2d(256),
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
        
        else:
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

original_model = Alexnet(10, False)
BN_model = Alexnet(10, True)


#dataset configuration
train_dataset = torchvision.datasets.CIFAR10(root = "../CIFAR_10", train=True, download=False, transform= transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))
test_dataset = torchvision.datasets.CIFAR10(root = "../CIFAR_10", train=False, download=False, transform= transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))

train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') #아마 label 선언이 없고 index로 나타내고 있음.

#Hyper parameters configuration
BN_optimizer = torch.optim.Adam(BN_model.parameters(), lr=0.001)
original_optimizer = torch.optim.Adam(original_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BN_model.to(DEVICE)
original_model.to(DEVICE)

summary(BN_model, input_size = (3, 224, 224), device = 'cuda')
summary(original_model, input_size = (3, 224, 224), device = 'cuda')


#visualization
BN_loss = []
original_loss = []
BN_acc = []
original_acc = []


def train(model, model_name, train_dataloader, optimizer, criterion, epoch, loss_visual):
    model.train()
    running_loss: float = 0.0
    
    #torch.backends.cudnn.benchmark = True
    
    for idx, (images, labels) in enumerate(tqdm(train_dataloader, desc=model_name, position=0, leave= False)):
        optimizer.zero_grad()
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        predict = model(images)
        loss = criterion(predict, labels)
        loss_visual.append(loss.item())
        loss.backward()
        

        optimizer.step()
        running_loss += loss.item()
       
    print('[%d, %5d] loss: %.4f' % (epoch + 1, idx + 1, running_loss / idx ))
    
    
def test(model, model_name, test_dataloader, criterion, acc):
    model.eval()
    correct = 0
    percent = 0.0
    loss_total = 0.0

    
    
    for idx, (images, labels) in enumerate(tqdm(test_dataloader, desc=model_name, position=0, leave = False)):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        predict = model(images)
        
        loss = criterion(predict, labels)
        predicted = torch.argmax(predict, dim = 1)
        correct += predicted.eq(labels).sum().item()
        
        
        loss_total += loss.item()

    acc.append(correct/ len(test_dataloader.dataset))
    percent = correct / len(test_dataloader.dataset)
    loss_total = loss_total / len(test_dataloader.dataset)
    print(f'test accuracy : {percent:.2f} \t avg_total : {loss_total:.4f}')


for epoch in range(EPOCHES):
    train(BN_model, "BN_Training", train_dataloader, BN_optimizer, criterion, epoch, BN_loss)
    train(original_model, "Original_Training", train_dataloader, original_optimizer, criterion, epoch, original_loss)
    test(BN_model, "BN_Testing", test_dataloader, criterion, BN_acc)
    test(original_model,"Original_Testing", test_dataloader, criterion, original_acc)


# BN_loss graph
plt.plot([i for i in range(len(BN_loss))], BN_loss)
plt.title('Train BN Loss')
plt.xlabel('step')
plt.ylabel('loss')
plt.show()

# original_loss graph
plt.plot([i for i in range(len(original_loss))], original_loss)
plt.title('Train original Loss')
plt.xlabel('step')
plt.ylabel('loss')
plt.show()

# BN_acc graph
plt.plot([i for i in range(len(BN_acc))], BN_acc)
plt.title('Test BN acc')
plt.xlabel('step')
plt.ylabel('acc')
plt.show()

# original_acc graph
plt.plot([i for i in range(len(original_acc))], original_acc)
plt.title('Test original acc')
plt.xlabel('step')
plt.ylabel('acc')
plt.show()