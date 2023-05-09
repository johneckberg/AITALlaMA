import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup, AdamW
from peft import get_peft_model, LoraConfig, TaskType

# Load your preprocessed training data into a dataset
# Assume your dataset class is `YourDataset` and loads data from `your_dataset_path`
train_dataset = YourDataset(your_dataset_path)

# Create a DataLoader for batching
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Load the model and configure it for PEFT
model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up the optimizer and learning rate scheduler
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training loop
model.train()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Prepare the inputs and move them to the appropriate device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Update the model parameters
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

# Save the fine-tuned model
model.save_pretrained("output_dir")

