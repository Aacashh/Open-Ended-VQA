import torch
from transformers import GPT2Tokenizer
from data_loader import get_loaders
from model import VQAModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_bleu(reference, hypothesis):
    """
    Compute the BLEU score between the reference and hypothesis.
    
    Args:
    - reference (str): The reference sentence.
    - hypothesis (str): The predicted sentence.

    Returns:
    - float: The BLEU score.
    """
    reference = reference.split()
    hypothesis = hypothesis.split()
    
    references = [reference]
    
    smoothing = SmoothingFunction().method1
    bleu = sentence_bleu(references, hypothesis, smoothing_function=smoothing)
    
    return bleu

def evaluate(model, dataloader, tokenizer, device):
    model.eval()
    total_bleu = 0.0

    with torch.no_grad():
        for batch in dataloader:
            images, questions, true_answers = batch
            images = images.to(device)
            questions = questions.to(device)

            generated_answers = model(images, questions)
            
            # Convert token IDs back to strings
            generated_answers = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_answers]

            # Compute BLEU for each pair
            for true, generated in zip(true_answers, generated_answers):
                total_bleu += compute_bleu(true, generated)

    average_bleu = total_bleu / len(dataloader.dataset)
    return average_bleu

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("cemilcelik/distilgpt2_pubmed")

    model_path = "./path_to_saved_model.pth"
    model = VQAModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    _, _, test_loader = get_loaders()

    bleu_score = evaluate(model, test_loader, tokenizer, device)
    print(f"Average BLEU score on the test set: {bleu_score:.4f}")
