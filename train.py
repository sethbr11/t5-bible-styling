import os
import pandas as pd
import torch
from datasets import Dataset, load_metric
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    logging as hf_logging
)
from tqdm import tqdm
from datasets.utils.logging import set_verbosity_error

# Clean up tqdm for nohup (line-by-line logging)
tqdm._instances.clear()
set_verbosity_error()
tqdm.monitor_interval = 0
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"

# Suppress future warnings from transformers
hf_logging.set_verbosity_error()

def preprocess_function(examples, tokenizer):
    prefix = "Convert to King James style: "
    inputs = [prefix + text if text is not None else "" for text in examples["modern_text"]]
    targets = [text if text is not None else "" for text in examples["kjv_text"]]

    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_preds):
    bleu = load_metric("bleu")
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.split() for pred in decoded_preds]
    decoded_labels = [[label.split()] for label in decoded_labels]
    result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["bleu"]}

def main():
    try:
        print("🔧 Loading model and tokenizer...")
        model_name = "t5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)

        print("📖 Loading dataset...")
        df = pd.read_csv("data/web_to_kjv.csv")
        dataset = Dataset.from_pandas(df)
        train_test_split = dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split["train"]
        val_dataset = train_test_split["test"]

        print("🧼 Preprocessing data...")
        tokenized_train = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
        tokenized_val = val_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        print("⚙️ Setting training parameters...")
        training_args = Seq2SeqTrainingArguments(
            output_dir="./t5-kjv-style",
            eval_strategy="epoch",
            learning_rate=5e-4,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            weight_decay=0.01,
            num_train_epochs=8,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            save_total_limit=2,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            logging_dir="./logs",
            logging_strategy="steps",
            logging_steps=50
        )

        print("🚀 Starting training...")
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        trainer.train()
        trainer.evaluate()

        print("💾 Saving model and tokenizer...")
        model.save_pretrained("./t5-kjv-style/fine_tuned_model")
        tokenizer.save_pretrained("./t5-kjv-style/fine_tuned_model")

        print("✅ Training complete!")

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user. Saving current model state...")
        model.save_pretrained("./t5-kjv-style/interrupt_save")
        tokenizer.save_pretrained("./t5-kjv-style/interrupt_save")
        print("✅ Model saved. Exiting cleanly.")

if __name__ == "__main__":
    main()
