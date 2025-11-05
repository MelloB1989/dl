import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader

transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),  # Changed to single values for mean and std
    ]
)

train_data = datasets.MNIST(
    root="./data_mnist", download=True, train=True, transform=transforms
)
test_data = datasets.MNIST(
    root="./data_mnist", download=True, train=False, transform=transforms
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


class Denoise_AE(nn.Module):
    def __init__(self):
        super(Denoise_AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 7),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = Denoise_AE()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(model)


def add_noise(img):
    noise = torch.randn(img.size()) * 0.5
    noise_img = img + noise
    noise_img = torch.clamp(noise_img, 0.0, 1.0)
    return noise_img


for epoch in range(20):
    for data in train_loader:
        img, _ = data
        img = img.to(device)
        noisy_img = add_noise(img)  # Add noise to the image
        output = model(noisy_img)
        loss = criterion(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch [{}/{}], loss:{:.4f}".format(epoch + 1, 20, loss.item()))

import numpy as np
import matplotlib.pyplot as plt
import torchvision


def imshow(img, cmap="gray"):
    npimg = img.numpy()
    plt.figure(figsize=(12, 6))
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap=cmap)
    plt.axis("off")
    plt.show()


show_img = torch.cat([img, noisy_img, output], 0)
show_img = show_img.cpu().detach()
print(show_img.shape)
imshow(torchvision.utils.make_grid(show_img), cmap="gray")
