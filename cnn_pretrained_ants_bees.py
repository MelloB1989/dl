import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Define Transformations
transforms = {
    "train": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
}

# Load the data
data_dir = "/content/dataset/hymenoptera_data"

image_datasets = {}
image_datasets["train"] = datasets.ImageFolder(
    os.path.join(data_dir, "train"), transform=transforms["train"]
)
image_datasets["val"] = datasets.ImageFolder(
    os.path.join(data_dir, "val"), transform=transforms["val"]
)
data_loader = {}
data_loader["train"] = DataLoader(image_datasets["train"], batch_size=32, shuffle=True)
data_loader["val"] = DataLoader(image_datasets["val"], batch_size=32, shuffle=True)

model = models.resnet50(pretrained=True)

model_ftrs = model.fc.in_features
model.fc = nn.Linear(model_ftrs, 2)
criterian = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)
print(model)

for epoch in range(10):
    for input, label in data_loader["train"]:
        output = model(input)
        loss = criterian(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(loss.item())

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for input, label in data_loader["val"]:
        output = model(input)
        _, predicted = torch.max(output, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    print(f"Accuracy : {(correct/total)*100}%")
print(total, correct)
