# CNN on CIFAR-10 using PyTorch (Simple and Clean)
# ------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 1️⃣ Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f"Using device: {device}")

# 2️⃣ Data loading and preprocessing
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert PIL images to tensors
        transforms.Normalize(
            (0.5, 0.5, 0.5),  # Normalize RGB channels
            (0.5, 0.5, 0.5),
        ),
    ]
)

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 3️⃣ Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # input: 3x32x32 → 32x32x32
        self.pool = nn.MaxPool2d(2, 2)  # 32x32x32 → 32x16x16
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 32x16x16 → 64x16x16
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleCNN().to(device)
print(model)

# 4️⃣ Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5️⃣ Training loop
num_epochs = 10
train_losses = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

print("Training complete.")

# 6️⃣ Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# 7️⃣ Visualize some predictions
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.show()


dataiter = iter(test_loader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images[:8]))
outputs = model(images.to(device))
_, preds = torch.max(outputs, 1)
print("Predicted:", " ".join(f"{classes[preds[j]]:5s}" for j in range(8)))
print("Actual:   ", " ".join(f"{classes[labels[j]]:5s}" for j in range(8)))
