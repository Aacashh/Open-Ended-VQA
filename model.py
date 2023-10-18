import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPProcessor, CLIPModel, GPT2LMHeadModel, GPT2Tokenizer
from typing import Tuple
from vision_encoder import VisionEncoder
from PIL import Image
class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(nn.Dropout(p=0.5))
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class VQAModel(nn.Module):
    def __init__(self):
        super(VQAModel, self).__init__()

        self.clip_model = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
        # self.clip_encode = CLIPImageEncoder.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("cemilcelik/distilgpt2_pubmed")
        self.gpt2_model = GPT2LMHeadModel.from_pretrained("cemilcelik/distilgpt2_pubmed")
                
        self.mapper = MLP(sizes=(512, 768, 768))

    def forward(self, images, questions):
        # preprocess = CLIPImageProcessor.from_pretrained('flaviagiammarino/pubmed-clip-vit-base-patch32')
        
        # image = preprocess(image, return_tensors="pt")
        # image_features = self.clip_model.encode_image(images)
        image_features = model.get_image_features(images)
        input_ids = self.gpt2_tokenizer(questions, return_tensors="pt", truncation=True, padding=True).input_ids
        question_features = self.gpt2_model.base_model(input_ids).last_hidden_state[:, 0, :]  # Extracting the [CLS] token's features

        combined_features = self.mapper(image_features) + question_features

        output = self.gpt2_model.generate(input_ids=None, max_length=50, encoder_outputs=(combined_features,))

        return output

if __name__ == '__main__':
    model = VQAModel()
    encoder = VisionEncoder()
    path = './data/img/synpic676.jpg'
    # sample_image_features = encoder([sample_image])
    sample_image = Image.open(path)
    sample_question = ["What does the area on the right side of the field show?"]
    output = model(sample_image, sample_question)
    answer = model.gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
    print(answer)