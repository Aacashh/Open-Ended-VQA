import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class TextEncoder(nn.Module):
    def __init__(self, pretrained_model: str = "cemilcelik/distilgpt2_pubmed"):
        super(TextEncoder, self).__init__()

        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        self.model = GPT2LMHeadModel.from_pretrained(pretrained_model)

    def tokenize(self, questions):
        """Tokenize the input questions."""
        return self.tokenizer(questions, return_tensors="pt", truncation=True, padding=True).input_ids

    def forward(self, questions):
        """Extract features from questions."""
        input_ids = self.tokenize(questions)
        features = self.model.base_model(input_ids).last_hidden_state[:, 0, :] 
        return features

if __name__ == '__main__':
    encoder = TextEncoder()
    sample_question = ["What does the area on the right side of the field show?"]
    
    features = encoder(sample_question)
    print(features.shape)  # torch.Size([1, 768])
