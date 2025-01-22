import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, get_scheduler
from torch.optim import AdamW
import wandb
import os
from model_1 import get_model

wandb.init(project="smollm-training", name="llama-smollm-corpus")

BATCH_SIZE = 8
SEQ_LEN = 256
LEARNING_RATE = 1e-4
EPOCHS = 5
WARMUP_STEPS = 1000
GRADIENT_CLIP_VAL = 1.0
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.7, top_k=50, device=DEVICE):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Apply top-k sampling
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            probs = torch.softmax(top_k_logits, dim=-1)
            
            # Sample from the filtered distribution
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices[0, next_token_idx[0]]
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    model.train()
    return generated_text

def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        # 'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'step': step
    }, path)

def load_checkpoint(path, model, optimizer, scheduler):
    if os.path.exists(path):
        # path = './checkpoints/checkpoint_step_5000.pt'
        # print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, weights_only=True)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['step']
    return 0, 0

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
if tokenizer.pad_token is None:
    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.resize_token_embeddings(len(tokenizer))

dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", streaming=True, split="train")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=SEQ_LEN, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True)


def collate_fn(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_loader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Initialize model, optimizer, and scheduler
model = get_model(tokenizer)
model.to(DEVICE)

# Print model parameters
# total_params = count_parameters(model)
# print(f"\nModel Statistics:")
# print(f"Total Parameters: {total_params:,}")
# print(f"Model Size: {total_params * 4 / (1024 * 1024):.2f} MB")  # Assuming float32 (4 bytes)
# print(f"Device: {DEVICE}")
# print(f"Batch Size: {BATCH_SIZE}")
# print(f"Sequence Length: {SEQ_LEN}")
# print(f"Learning Rate: {LEARNING_RATE}")
# print("-" * 50 + "\n")

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    total_steps=10000,
    pct_start=0.1,
    anneal_strategy="cos",
    cycle_momentum=False
)

# Load checkpoint if exists
start_epoch, global_step = load_checkpoint(
    f"{CHECKPOINT_DIR}/latest_checkpoint.pt",
    model,
    optimizer,
    lr_scheduler
)

# Sample prompts for evaluation
sample_prompts = [
    "The future of artificial intelligence",
    "The most important thing in life",
    "The best way to learn programming"
]

model.train()
try:
    for epoch in range(start_epoch, EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        for step, batch in enumerate(train_loader, start=global_step):
            # Move batch to device
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # Forward pass
            outputs = model(input_ids)
            logits = outputs.view(-1, tokenizer.vocab_size)
            
            # Calculate loss with label smoothing
            loss = torch.nn.functional.cross_entropy(
                logits, 
                labels.view(-1),
                label_smoothing=0.1  # Add label smoothing
            )

            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
            optimizer.step()
            lr_scheduler.step()

            # Logging
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}, LR: {lr_scheduler.get_last_lr()[0]:.2e}")
                wandb.log({
                    "loss": loss.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": step,
                    "epoch": epoch
                })

            # Save checkpoint every 100 steps
            if step % 100 == 0:
                save_checkpoint(
                    model, optimizer, lr_scheduler,
                    epoch, step, loss.item(),
                    f"{CHECKPOINT_DIR}/latest_checkpoint.pt"
                )
                
                # Also save numbered checkpoint every 1000 steps
                if step % 1000 == 0:
                    save_checkpoint(
                        model, optimizer, lr_scheduler,
                        epoch, step, loss.item(),
                        f"{CHECKPOINT_DIR}/checkpoint_step_{step}.pt"
                    )

            # Generate sample text every 500 steps with different temperatures
            if step % 500 == 0:
                print("\n=== Generating Sample Texts ===")
                for temp in [0.7, 1.0]:  # Try different temperatures
                    for prompt in sample_prompts:
                        generated = generate_text(
                            model, 
                            tokenizer, 
                            prompt, 
                            temperature=temp,
                            max_length=100  # Increased max length
                        )
                        print(f"\nPrompt: {prompt}")
                        print(f"Temperature: {temp}")
                        print(f"Generated: {generated}")
                        wandb.log({
                            f"generated_text_temp_{temp}_{prompt[:20]}": wandb.Html(generated)
                        })
                print("\n=== End of Samples ===\n")
                model.train()

        # Save epoch checkpoint
        save_checkpoint(
            model, optimizer, lr_scheduler,
            epoch, step, loss.item(),
            f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch+1}.pt"
        )

except KeyboardInterrupt:
    print("\nTraining interrupted! Saving checkpoint...")
    save_checkpoint(
        model, optimizer, lr_scheduler,
        epoch, step, loss.item(),
        f"{CHECKPOINT_DIR}/interrupted_checkpoint.pt"
    )

print("Training complete!")
wandb.finish()


