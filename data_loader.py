
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class VQARADDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.vqa_rad_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.vqa_rad_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.vqa_rad_frame.iloc[idx, 0])
        image = Image.open(img_name)
        
        question = self.vqa_rad_frame.iloc[idx, 1]
        answer = self.vqa_rad_frame.iloc[idx, 2]
        
        sample = {'image': image, 'question': question, 'answer': answer}
        
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        
        return sample

def get_loaders(csv_file='./data/vqa_rad.csv', img_dir='./data/img/', batch_size=32, transform=None, split_ratio=(0.8, 0.1, 0.1)):
    """
    Returns training, validation, and test data loaders.
    Args:
        csv_file (string): Path to the csv file.
        img_dir (string): Directory with all the images.
        batch_size (int): Batch size for DataLoader.
        transform (callable, optional): Optional transform to be applied on image.
        split_ratio (tuple): Ratios for train, val, and test split. They should sum to 1.
    """

    assert sum(split_ratio) == 1, "Split ratios should sum to 1."

    dataset = VQARADDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    
    total_size = len(dataset)
    train_size = int(split_ratio[0] * total_size)
    val_size = int(split_ratio[1] * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])
    train_loader, val_loader, test_loader = get_loaders(transform=transform)

    for batch in train_loader:
        # Example of how to access the data in train_loader
        images, questions, answers = batch['image'], batch['question'], batch['answer']
        
        # Just printing the shape and first item to see if everything is working correctly
        print(images.shape)
        print(questions[0])
        print(answers[0])
        break