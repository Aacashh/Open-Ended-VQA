import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_loaders
from model import VQAModel

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
MODEL_PATH = './models/best_model.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, val_loader, _ = get_loaders(batch_size=BATCH_SIZE)

model = VQAModel().to(device)

criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, questions, answers in dataloader:
        images = images.to(device)
        questions = questions.to(device)
        answers = answers.to(device)
        
        optimizer.zero_grad()
        outputs = model(images, questions)
        loss = criterion(outputs, answers)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, questions, answers in dataloader:
            images = images.to(device)
            questions = questions.to(device)
            answers = answers.to(device)

            outputs = model(images, questions)
            _, predicted = outputs.max(1)
            total += answers.size(0)
            correct += predicted.eq(answers).sum().item()
            
            loss = criterion(outputs, answers)
            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

best_val_loss = float('inf')
for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = validate(model, val_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/{EPOCHS} => Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val accuracy: {val_accuracy:.2f}%")
  
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

print("Training complete.")
