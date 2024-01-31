import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from helpers import tokenize, generate_prompt
from datetime import datetime

USERNAME = "0xideas"
DATA_PATH = "../data"
DATA_FILE = "corpus.jsonl"

PAD_TOKEN = "</s>"
CUTOFF_LEN = 2048  #Our dataset has shot text
LORA_R = 8
LORA_ALPHA = 2 * LORA_R
LORA_DROPOUT = 0.1


now = datetime.now()
BASE_MODEL_SOURCE = "mistralai"
BASE_MODEL = "Mistral-7B-v0.1"
MODEL_VARIANT = f"rust-instruct"
MODEL_SUBVARIANT = f"{DATA_FILE.replace('rust-corpus-', '').replace('.jsonl', '')}"
MODEL_VARIANT_TS = f"{MODEL_VARIANT}-{MODEL_SUBVARIANT}-{now.year}-{now.month}-{now.day}-{now.hour}"

input(f"hugging face repo: {MODEL_VARIANT}")
input(f"full model name: {MODEL_VARIANT_TS}")



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

    # config = LoraConfig(
    #    r=LORA_R,
    #    lora_alpha=LORA_ALPHA,
    #    target_modules=[ "w1", "w2", "w3"],  #just targetting the MoE layers.
    #    lora_dropout=LORA_DROPOUT,
    #    bias="none",
    #    task_type="CAUSAL_LM"
    #)

    config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
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
            num_train_epochs=3,
            learning_rate=1e-4,
            logging_steps=2,
            optim="adamw_torch",
            save_strategy="epoch",
            output_dir=MODEL_VARIANT
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    model.config.use_cache = False

    trainer.train()

    model_output_path = f"{USERNAME}/{MODEL_VARIANT_TS}"
    print(f"{model_output_path = }")
    trainer.push_to_hub(model_output_path)