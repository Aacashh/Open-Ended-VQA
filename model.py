import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, GPT2LMHeadModel, GPT2Tokenizer
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
                layers.append(nn.Dropout(p=0.3))
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class VQAModel(nn.Module):
    def __init__(self):
        super(VQAModel, self).__init__()

        self.clip_model = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
        self.gpt2_model = GPT2LMHeadModel.from_pretrained("cemilcelik/distilgpt2_pubmed")
        self.tokenizer = GPT2Tokenizer.from_pretrained("cemilcelik/distilgpt2_pubmed")
        self.mapper = MLP(sizes=(512, 768, 768)) 
        self.project_down = nn.Linear(1280, 768)

    def forward(self, images, questions):
        image_features = self.clip_model.get_image_features(images)
        question_features = self.gpt2_model.base_model(questions).last_hidden_state[:, 0, :]
#         print("image_features shape:", image_features.shape)
#         print("question_features shape:", question_features.shape)
        combined_features = torch.cat((image_features, question_features), dim=-1)
        combined_features = self.project_down(combined_features)
        combined_features = combined_features.unsqueeze(1) 
        outputs = self.gpt2_model(inputs_embeds=combined_features)
        logits = outputs.logits
        eos_token_id = self.tokenizer.encode("<END>", add_prefix_space=True)[0]
        generated_sequence = self.gpt2_model.generate(inputs_embeds=combined_features, 
                                                      max_length=36, 
                                                      pad_token_id=self.gpt2_model.config.pad_token_id, 
                                                      repetition_penalty=1, 
                                                      eos_token_id=eos_token_id)
        return logits, generated_sequence
    
if __name__ == '__main__':
    model = VQAModel()
    encoder = VisionEncoder()
    path = './data/img/synpic676.jpg'
    sample_image = Image.open(path)
    sample_question = ["What does the area on the right side of the field show?"]
    output = model(sample_image, sample_question)
    answer = model.gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
    print(answer)