from transformers import ViTForImageClassification
from transformers import AutoImageProcessor
from datasets import load_dataset

import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader

from data.CustomData import CustomDataset

# Preparing the data
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

training_data = load_dataset("microsoft/cats_vs_dogs")
training_dataset = CustomDataset(training_data["train"], transform)
train_loader = DataLoader(training_dataset, batch_size=32, shuffle=True)

model = ViTForImageClassification.from_pretrained("facebook/deit-tiny-patch16-224")
model.classifier = nn.Linear(model.classifier.in_features, 2)
processor = AutoImageProcessor.from_pretrained("facebook/deit-tiny-patch16-224")

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
        inputs = processor(images, return_tensors="pt").to(device)  # Preprocess images
        outputs = model(**inputs).logits

        optimizer.zero_grad()
        # outputs = model(images).logits  # Get logits from model
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

print("Fine-tuning complete!")

model.save_pretrained("fine_tuned_deit")

