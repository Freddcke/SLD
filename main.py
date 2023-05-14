import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from wlasldataset import WLASL100
from torch.utils.data import dataloader
import torch.optim as optim
from resnet import generate_model
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

num_classes = 25

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layer1 = self._conv_layer_set(3, 128)
        self.conv_layer2 = self._conv_layer_set(128, 256)
        self.conv_layer3 = self._conv_layer_set(256, 512)
        self.fc1 = nn.Linear(12*12*1*512, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.LeakyReLU()
        self.batch=nn.BatchNorm1d(128)
        self.drop=nn.Dropout(p=0.15)        
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    
    def forward(self, x):
        '''
        Forward takes a 5D tensor (a stack of RGB images) and produces a set of 25 one-hot encoded class labels
        '''
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out


def main():
    writer = SummaryWriter()

    train_dataset = WLASL100('./datafiles/alphatrain.csv', './videos', image_transform)
    test_dataset = WLASL100('./datafiles/alphatest.csv', './videos', image_transform)
    train_loader = dataloader.DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=8, drop_last=True)
    test_loader = dataloader.DataLoader(test_dataset, batch_size=6, num_workers=8, drop_last=True)
    ''' 
    Generate model is used for 3DResnet which is used as a comparison for performance
    '''
    net = generate_model(50).to(device)#CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-6)
    '''
    Schedular lowers learning rate over a few epochs, however this does not help the learning for this model,
    I believe this is due to the learning rate already being so low. It has been kept here as it is good practice
    to use but is not needed here.
    '''
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    for epoch in range(40):  
        train(net, criterion, optimizer, train_loader, epoch, writer)
        test(net, test_loader, epoch, writer)
        #scheduler.step()
        #torch.save(net.state_dict(), "checkpoint.pth")


def train(net, criterion, optimizer, trainloader, epoch, writer):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs = data[0].to(device)
        labels = data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('[%d, %5d] loss: %.6f' % (epoch, i, running_loss / i))
    writer.add_scalar("Loss/train", (running_loss / i), epoch)

def test(net, testloader, epoch, writer):
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs = data[0].to(device)
            labels = data[1].to(device)
            outputs = net(inputs)
            _, pred = torch.max(outputs.data, 1)
            targ = torch.argmax(labels.data, 1)
            total += labels.size(0)
            correct += (pred.float() == targ.float()).sum().item()
    print(f'Accuracy: {100 * correct // total} %')
    writer.add_scalar("Accuracy/test", (100 * correct // total), epoch)

if __name__ == '__main__':
    main()
