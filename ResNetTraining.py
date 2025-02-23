from torchvision import transforms
from transformers import ResNetForImageClassification
from transformers import AutoConfig, AutoModel
from datasets import load_dataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from data.CustomData import CustomDataset


# Step 1: Setting up the dataset using the cats-dogs dataset from Microsoft using HuggingFace API


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

## Step 1.1: defining Custom Dataset
train_data = load_dataset("microsoft/cats_vs_dogs")
train_dataset = CustomDataset(dataset=train_data["train"], transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

## Preparing ResNet architecture and training
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
model.config.num_labels = 2

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3)  # Train only classifier

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).logits  # Get logits from model
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

print("Fine-tuning complete!")

model.save_pretrained("fine_tuned_resnet")
