import torch.nn as nn
import torch
from transformers import CLIPModel
from torchvision import transforms

class VisionEncoder(nn.Module):
    def __init__(self, pretrained_model: str = "flaviagiammarino/pubmed-clip-vit-base-patch32"):
        super(VisionEncoder, self).__init__()

        # Load the CLIP model and tokenizer
        self.model = CLIPModel.from_pretrained(pretrained_model)
        self.transform = self.get_image_transforms()

    def get_image_transforms(self):
        """Defines the image preprocessing pipeline."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])

    def forward(self, images):
        """Extract features from images."""
        images = [self.transform(image) for image in images]
        images = torch.stack(images, dim=0)
        features = self.model.encode_image(images)
        return features

if __name__ == '__main__':
    encoder = VisionEncoder()
    from PIL import Image
    sample_image = Image.open("./data/img/synpic676.jpg")
    features = encoder([sample_image])
    print(features.shape)  # Expected: torch.Size([1, 512])
