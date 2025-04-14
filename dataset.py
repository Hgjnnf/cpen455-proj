import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize
from bidict import bidict
from tqdm import tqdm
import pandas as pd

# Rescaling functions
rescaling = lambda x: (x - 0.5) * 2.
rescaling_inv = lambda x: 0.5 * x + 0.5
replicate_color_channel = lambda x: x.repeat(3, 1, 1)

# Bidict mapping (if needed)
my_bidict = bidict({'Class0': 0, 'Class1': 1, 'Class2': 2, 'Class3': 3})

class CPEN455Dataset(Dataset):
    def __init__(self, root_dir='./data', mode='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        csv_path = os.path.join(self.root_dir, mode + '.csv')
        df = pd.read_csv(csv_path, header=None, names=['path', 'label'])
        # Create list of tuples: (full_image_path, numeric label)
        self.samples = [(os.path.join(self.root_dir, path), int(label))
                        for path, label in df.itertuples(index=False)]
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = read_image(img_path).float() / 255.0
        if image.shape[0] == 1:
            image = replicate_color_channel(image)
        if self.transform:
            image = self.transform(image)
        return image, label

def show_images(images, categories, mode: str):
    fig, axs = plt.subplots(1, len(images), figsize=(15, 5))
    for i, image in enumerate(images):
        axs[i].imshow(image.permute(1, 2, 0))
        axs[i].set_title(f"Category: {categories[i]}")
        axs[i].axis('off')
    plt.savefig(mode + '_test.png')

if __name__ == '__main__':
    transform_32 = Compose([
        Resize((32, 32)),
        rescaling
    ])
    dataset_list = ['train', 'validation', 'test']
    for mode in dataset_list:
        print(f"Mode: {mode}")
        dataset = CPEN455Dataset(root_dir='./data', transform=transform_32, mode=mode)
        data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
        for images, categories in tqdm(data_loader):
            print(images.shape, categories)
            images = torch.round(rescaling_inv(images) * 255).type(torch.uint8)
            show_images(images, categories, mode)
            break
