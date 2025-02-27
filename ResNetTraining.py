from torch.utils.data import DataLoader
from transformers import ResNetForImageClassification, AutoFeatureExtractor
import torch
from tqdm import tqdm

from datasets import load_dataset

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        self.label_map = {"cat":0, "dog":1}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index]["image"]
        label = self.dataset[index]["labels"]
        image = image.convert('RGB') if hasattr(image, 'convert') else image
        image = self.transform(image, return_tensors="pt")
        if len(image['pixel_values'].shape) > 3:
            image['pixel_values'] = image['pixel_values'].squeeze(0)
            #print("squeezing")
        return image, label


# Instantiating the Model
model_name = "microsoft/resnet-50"
model = ResNetForImageClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# Preparing the datasets
raw_data = load_dataset("microsoft/cats_vs_dogs")
train_dataset = CustomDataset(raw_data["train"], feature_extractor)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 2)

device = torch.device("cuda")
model.to(device)


# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=10e-5)

#for epoch in range(1):
#    for (image, label) in train_dataloader:
#        print(type(image), type(label))

epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for (image, label) in tqdm(train_dataloader, total=len(train_dataloader)):
        image = image.to(device)
        label = label.to(device)

        model.zero_grad()
        pred = model(**image)
        loss = criterion(pred.logits, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch: {epoch} total_loss: {total_loss}")
print("Finished Fine Tuning!!")
