from local import ReLULocalZeroToken
import re, os
import argparse
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorWithFlattening
from datasets import load_dataset
import torch

sclass = ReLULocalZeroToken

def setmodule(module: nn.Module, target_module: str, value: nn.Module):
    """Set a target module from in a given module."""
    submodules = target_module.split(".", 1)
    if len(submodules) == 1:
        if submodules[0].isdigit():
            module[int(submodules[0])] = value
        else:
            setattr(module, submodules[0], value)
    else:
        setmodule(getattr(module, submodules[0]), submodules[-1], value)

def set_zerotoken_layers(llm):
    for name, layer in llm.named_modules():
        if re.search(r"layers\.\d+$", name):
            setmodule(llm, name, sclass(layer, llm.config))

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with NaiveLocalZeroToken layers")
    
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM3-3B",
                        help="Model name or path")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceTB/smoltalk2",
                        help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="./zero-token-model",
                        help="Output directory for checkpoints")
    
    return parser.parse_args()

def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16)
    
    set_zerotoken_layers(model)

    dataset = load_dataset(args.dataset_name, split="train")
    # dataset = dataset.filter(lambda x: x["source"] == "smollm3_smol-magpie-ultra", num_proc=8)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(50_000))
    dataset = dataset.map(lambda x: {"templated": tokenizer.apply_chat_template(
        x["messages"], tokenize=False, add_generation_prompt=False)}, num_proc=os.cpu_count())
    tokenized_datasets = dataset.map(lambda x: tokenizer(x["templated"]), num_proc=os.cpu_count())
    tokenized_datasets = tokenized_datasets.map(lambda x: {"length": len(x["input_ids"])}, num_proc=os.cpu_count())
    tokenized_datasets = tokenized_datasets.filter(lambda x: x["length"] <= 1024, num_proc=os.cpu_count())
    tokenized_datasets = tokenized_datasets.sort("length")
    print(tokenized_datasets)
        
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        # grad_norm=1,
        # eval_steps=500,
        save_steps=100_000,
        learning_rate=5e-5,
        bf16=True,
        report_to="none",
        remove_unused_columns=True
    )

    class ModelWraper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.idx = 0

        def forward(self, input_ids, attention_mask):
            labels = input_ids.clone()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            reg_loss = 0.0
            total_losses = 0.0
            mean_zero_tokens = 0.0
            for module in self.model.modules():
                if isinstance(module, sclass) and module.training:
                    reg_loss += module.reg_loss
                    total_losses += 1.0
                    mean_zero_tokens += module.mean_zero_tokens

            self.idx += 1
            if self.idx % 100 == 0:
                print(f"Step {self.idx}: LM Loss = {outputs.loss.item()}, Mean Zero Tokens = {mean_zero_tokens / total_losses}")
            outputs.loss = outputs.loss + (reg_loss / total_losses)
            return outputs

    model = ModelWraper(model)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )
    
    # Train
    print(model)
    print("Starting training...")
    trainer.train()
    
    # Save model
    final_output_dir = f"{args.output_dir}-final"
    print(f"Saving final model to {final_output_dir}")
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

if __name__ == "__main__":
    main()

# module purge
# module load arch
# ...