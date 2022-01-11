import torch
import torch.nn as nn

import torchvision
import torch.functional as F
import torchsummary as summary
import torchvision.models as models

from torchvision.transforms import transforms

BATCH_SIZE = 32
EPOCHS = 10
PRETRAINED = True
EXTRACT = True  #True : Feature_Extraction / False : Fine-Tuning
#MODEL
train_dataset = torchvision.datasets.CIFAR10(root = "../CIFAR_10", train=True, download=False, transform= transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
test_dataset = torchvision.datasets.CIFAR10(root = "../CIFAR_10", train=False, download=False, transform= transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))

train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle=False)

model = models.resnet18(pretrained = PRETRAINED)

#Feature_Extraction
def set_requires_grad(model, extract):
    if extract: #True
        for params in model.parameters():
            params.requires_grad = False    

#Feature_Extract         
set_requires_grad(model, EXTRACT)
num_ftrs = model.fc.in_features #512
num_classes = 10
model.fc = torch.nn.Linear(num_ftrs, num_classes)

#GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#? optimizer setting
model = model.to(device)
params_to_update = model.parameters()
print("Parmas to learn : ")

if EXTRACT: #FeaTrue Extraction
    params_to_update = []
    for name, params in model.named_parameters():
        if params.requires_grad == True:
            params_to_update.append(params)
            print("\t", name)
    optimizer_ft = torch.optim.Adam(params_to_update, 0.001)

else:   #Fine-Tuning. same setting, but fine-tuning has that all parameters requires_grad is True
    for name, params in model.named_parameters():
        if params.requires_grad == True:
            print("\t", name)
            
    #adapt differnt lr parameters // condition : Fine Tune Transfer-Learning
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    fc = []
    for n, p in model.named_parameters():
        if 'layer1' in n:
            l1.append(p)
        elif 'layer2' in n:
            l2.append(p)
        elif 'layer3' in n:
            l3.append(p)
        elif 'layer4' in n:
            l4.append(p)
        else:
            fc.append(p)
    lr_update = [{'params' : l1, 'lr' : 1e-4},
                {'params' : l2, 'lr' : 5e-4},
                {'params' : l3, 'lr' : 1e-3},
                {'params' : l4, 'lr' : 5e-4},
                {'params' : fc, 'lr' : 1e-5}]
    optimizer_ft = torch.optim.Adam(lr_update)

#setup the loss
criterion = nn.CrossEntropyLoss()

#Training..
def train(epochs, model, train_dataloader, optimizer, criterion):
    model.train()
    train_loss = 0
    corrects = 0
    train_acc = 0
    
    
    torch.backends.cudnn.benchmark = True
    
    for idx, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(output, 1)
        corrects += torch.sum(preds == labels.data)
        
        if idx % 500 == 0:
            print(f'idx : {idx} \t loss_value : {loss:.4f} \t')
    
    #Epochs Loss
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    print(f'epochs : {epochs} \t final loss : {train_loss:.4f} acc : {train_acc:.2f}')
    
    
for epoch in range(EPOCHS):
    train(epoch, model, train_dataloader, optimizer_ft, criterion)
    
#summary.summary(model, (3, 32, 32), batch_size= 16, device='cpu')