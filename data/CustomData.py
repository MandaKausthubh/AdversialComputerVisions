import torch

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
        image = self.transform(image)
        return image, label
