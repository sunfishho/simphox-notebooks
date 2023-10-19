import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchonn as onn
from torchonn.models import ONNBaseModel
import torch.optim as optim
import torchvision.transforms as transforms

class ONNModel(ONNBaseModel):
    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        self.conv1 = onn.layers.MZIBlockConv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=True,
            miniblock=4,
            mode="usv",
            decompose_alg="clements",
            photodetect=True,
            device=device,
        )
        self.conv2 = onn.layers.MZIBlockConv2d(
            in_channels=6,
            out_channels=10,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=True,
            miniblock=4,
            mode="usv",
            decompose_alg="clements",
            photodetect=True,
            device=device,
        )
        self.pool = nn.AdaptiveAvgPool2d(5)
        self.linear = onn.layers.MZIBlockLinear(
            in_features=10*5*5,
            out_features=10,
            bias=True,
            miniblock=4,
            mode="usv",
            decompose_alg="clements",
            photodetect=True,
            device=device,
        )
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the data
])

cifar_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                    download=True, transform=transform)
cifar_trainloader = torch.utils.data.DataLoader(cifar_trainset, batch_size=64,
                                        shuffle=True, num_workers=0)

cifar_testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)
cifar_testloader = torch.utils.data.DataLoader(cifar_testset, batch_size=64,
                                        shuffle=False, num_workers=0)
dtype = torch.float32

def cifar_check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on train set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

def cifar_train(model, optimizer, epochs=10):
    """
    Train a model on CIFAR using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(cifar_trainloader):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # to avoid explosions that create NaNs
            nn.utils.clip_grad_norm_(model.parameters(), 1)

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

        print(f'Epoch {e}, loss = {loss.item()}')
        if e % 5 == 0:
            test_accuracies.append(cifar_check_accuracy(cifar_testloader, model))
        if e % 10 == 0:
            train_accuracies.append(cifar_check_accuracy(cifar_trainloader, model))
        print()

if __name__ == '__main__':
    test_accuracies = []
    train_accuracies = []

    device = torch.device("cpu")
    cifar_model = ONNModel()
    learning_rate = 0.001

    optimizer = optim.SGD(cifar_model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    cifar_train(cifar_model, optimizer, epochs = 50)
