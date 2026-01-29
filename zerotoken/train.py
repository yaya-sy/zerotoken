from local import ReLULocalZeroToken
import importlib
import re, os
import argparse
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithFlattening
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
    parser = argparse.ArgumentParser(description="Train a model with ZeroToken sparsity")
    
    parser.add_argument("--model_name", "-m", type=str, default="HuggingFaceTB/SmolLM3-3B",
                        help="Model name or path")
    parser.add_argument("--dataset_name", "-d", type=str, default="HuggingFaceTB/smoltalk2",
                        help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="./zero-token-model",
                        help="Output directory for checkpoints")
    parser.add_argument("--approach", type=str, default="local", choices=["local", "global"],
                        help="Sparsity approach: apply per-layer (local) or before the LLM (global)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    global sclass

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                 torch_dtype=torch.bfloat16,
                                                 attn_implementation = "flash_attention_2")
    
    if args.approach == "local":
        sclass = ReLULocalZeroToken
        set_zerotoken_layers(model)
    else:
        global_mod = importlib.import_module("global")
        sclass = global_mod.ReLUGlobalZeroToken
        model = sclass(model)
        print(model)
    
    data_collator = DataCollatorWithFlattening(
        return_flash_attn_kwargs=True,
    )

    dataset = load_dataset(args.dataset_name, split="train")
    # dataset = dataset.filter(lambda x: x["source"] == "smollm3_smol-magpie-ultra", num_proc=8)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(100_000))
    dataset = dataset.map(lambda x: {"templated": tokenizer.apply_chat_template(
        x["messages"], tokenize=False, add_generation_prompt=False)}, num_proc=8)
    tokenized_datasets = dataset.map(lambda x: tokenizer(x["templated"]), num_proc=8)
    tokenized_datasets = tokenized_datasets.map(lambda x: {"length": len(x["input_ids"])}, num_proc=8)
    tokenized_datasets = tokenized_datasets.filter(lambda x: x["length"] <= 1024, num_proc=8)
    tokenized_datasets = tokenized_datasets.sort("length")
    print(tokenized_datasets)
        
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
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
        remove_unused_columns=False
    )

    class ModelWraper(nn.Module):
        def __init__(self, model, reg_module_cls):
            super().__init__()
            self.model = model
            self.idx = 0
            self.reg_module_cls = reg_module_cls

        def forward(self, *args, **kwargs):
            outputs = self.model(*args, **kwargs)

            device = outputs.loss.device
            reg_loss_sum = torch.zeros((), device=device, dtype=outputs.loss.dtype)
            # w_l2_sq_sum  = torch.zeros((), device=device, dtype=outputs.loss.dtype)
            mean_zero_tokens_sum = torch.zeros((), device=device, dtype=outputs.loss.dtype)

            count = 0

            for module in self.model.modules():
                if isinstance(module, self.reg_module_cls) and module.training:
                    count += 1

                    if module.reg_loss is not None:
                        reg_loss_sum += torch.as_tensor(
                                module.reg_loss, device=device, dtype=outputs.loss.dtype
                            )

                    if module.mean_zero_tokens is not None:
                        mean_zero_tokens_sum += torch.as_tensor(
                                module.mean_zero_tokens, device=device, dtype=outputs.loss.dtype
                            )

                    # w_l2_sq_sum += sum(m.weight.float().pow(2).sum().to(outputs.loss.dtype) for m in module.sparsifyer if isinstance(m, nn.Linear))

            reg_loss_mean = reg_loss_sum / count if count else reg_loss_sum
            # w_l2_sq_mean = w_l2_sq_sum / count
            mean_zero_tokens = mean_zero_tokens_sum / count if count else mean_zero_tokens_sum
            # else:
            # reg_loss_mean = self.model.reg_loss
            # mean_zero_tokens = self.model.mean_zero_tokens
            self.idx += 1
            if self.idx % 100 == 0:
                print(
                    f"Step {self.idx}: LM Loss = {outputs.loss.item():.6f}, "
                    f"Mean Zero Tokens = {mean_zero_tokens.item():.6f}, Reg Loss = {reg_loss_mean.item():.6f}"
                    # f"L2sq(mean) = {w_l2_sq_mean.item():.6f}"
                )

            outputs.loss = outputs.loss + reg_loss_mean #  + 1e-6 * w_l2_sq_mean

            return outputs

        def save_pretrained(self, *args, **kwargs):
            return self.model.save_pretrained(*args, **kwargs)

    model = ModelWraper(model, sclass)
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
