import torch
import random
from transformers import AutoTokenizer, LlamaConfig
from datasets import load_dataset
from modified_llama import ModifiedLlamaForCausalLM
from transformers import get_scheduler
import functools

device = torch.device( 'cuda' ) if torch.cuda.is_available() else torch.device( 'cpu' )

def preprocess_data(example, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(b["input_ids"]) for b in batch])
    attention_mask = torch.stack([torch.tensor(b["attention_mask"]) for b in batch])
    labels = input_ids.clone()

    # Generate a random flag for the entire batch
    flag = random.choice(['s', 'm', 'l', 'xl'])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "flag": flag}

def evaluate_model(model, eval_dataloader, flags):
    """Evaluate the model on the eval dataset for each flag and return losses."""
    model.eval()
    eval_losses = {flag: 0.0 for flag in flags}
    num_batches = len(eval_dataloader)

    with torch.no_grad():
        for flag in flags:
            total_loss = 0.0
            for batch in eval_dataloader:
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                labels = batch["labels"].to(model.device)

                # Configure the subnetwork for the flag
                model.configure_subnetwork(flag)

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()

            eval_losses[flag] = total_loss / num_batches

    model.train()
    return eval_losses

if __name__ == "__main__":
    # Load tokenizer
    print("loading tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset and preprocess
    print("loading dataset")
    dataset = load_dataset("vilm/RedPajama-v2-small", split="train")
    # Shuffle the dataset to ensure randomness
    dataset = dataset.shuffle(seed=42)
    # Select the first 100,000 examples
    dataset = dataset.select(range(10000))
    # map over the dataset and transform.
    print("preprocessing dataset")
    dataset = dataset.map(functools.partial(preprocess_data, tokenizer=tokenizer), num_proc=32)

    # Initialize Llama configuration and model from scratch
    print("loading config", flush=True)
    config = LlamaConfig.from_pretrained("NousResearch/Llama-3.2-1B")
    print("initializing model. This may take a while... ", end="", flush=True)
    model = ModifiedLlamaForCausalLM(config).to(device)
    print("Done!")

    # Split dataset into train and evaluation
    batch_size = 8
    eval_dataset = dataset.select(range(20 * batch_size))  # 10 batches for evaluation
    train_dataset = dataset.select(range(20 * batch_size, len(dataset)))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Training arguments
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Define the number of training steps
    num_training_steps = 10000  # 200 warmup, remaining cosine decay
    num_warmup_steps = 200

    # Define the scheduler
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    model.train()

    flags = ['s', 'm', 'l', 'xl']
    step = 0

    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)
        flag = batch["flag"]

        # Configure the subnetwork for the entire batch
        model.configure_subnetwork(flag)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step the scheduler
        scheduler.step()

        step += 1
        print(f"Step {step}, Loss: {loss.item()}")

        # Evaluate every 100 steps
        if step % 100 == 0:
            eval_losses = evaluate_model(model, eval_dataloader, flags)
            print(f"Step {step}, Eval Losses: {eval_losses}")

        # Stop training after the defined number of steps
        if step >= num_training_steps:
            break
