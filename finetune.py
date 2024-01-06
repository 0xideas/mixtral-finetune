import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from helpers import tokenize, generate_prompt

DATA_PATH = "/media/leon/2800931A0092EE56"
DATA_FILE = "rust-corpus-1k.jsonl"

PAD_TOKEN = "</s>"
CUTOFF_LEN = 256  #Our dataset has shot text
LORA_R = 8
LORA_ALPHA = 2 * LORA_R
LORA_DROPOUT = 0.1

BASE_MODEL_SOURCE = "mistralai"
BASE_MODEL = "Mistral-7B-v0.1"
MODEL_VARIANT = "test"



if __name__ == "__main__":

    # mistralai/Mixtral-8x7B-Instruct-v0.1
    tokenizer = AutoTokenizer.from_pretrained(f"{BASE_MODEL_SOURCE}/{BASE_MODEL}")
    tokenizer.pad_token = PAD_TOKEN
    model = AutoModelForCausalLM.from_pretrained(f"{BASE_MODEL_SOURCE}/{BASE_MODEL}",
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[ "w1", "w2", "w3"],  #just targetting the MoE layers.
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)

    dataset = load_dataset('json', data_files=f'{DATA_PATH}/{DATA_FILE}')

    print("dataset", dataset)
    train_data = dataset["train"] # Not using evaluation data

    train_data = train_data.shuffle().map(lambda x: tokenize(tokenizer, generate_prompt(x), CUTOFF_LEN), remove_columns=["instruction" , "input", "output"])

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=6,
            learning_rate=1e-4,
            logging_steps=2,
            optim="adamw_torch",
            save_strategy="epoch",
            output_dir="mixtral-moe-lora-instruct-shapeskeare"
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    model.config.use_cache = False

    trainer.train()

    trainer.push_to_hub(f"leontl/{BASE_MODEL}-{MODEL_VARIANT}")