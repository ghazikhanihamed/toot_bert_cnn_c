import os
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from settings import settings
from classes.PLMDataset import PLMDataset
import json

# Define the tasks and corresponding models
tasks = {"IC-MP": settings.ESM1B, "IT-MP": settings.ESM1B, "IC-IT": settings.ESM2}

datasets = {
    "IC-MP": settings.IC_MP_Train_DATASET,
    "IT-MP": settings.IT_MP_Train_DATASET,
    "IC-IT": settings.IC_IT_Train_DATASET,
}

# Path to your datasets
path = "./dataset/feb112024/raw/"

# Folders representing each class
folders = ["ionchannels", "iontransporters", "membrane_proteins"]

# We load the dictionary to map the labels which are sequence ids to a number
with open(settings.SEQUENCE_IDS_DICT_PATH, "r") as f:
    seqid_dict = json.load(f)

# Iterate over tasks and their corresponding models
for task, model_info in tasks.items():
    print(f"Task: {task}")
    print(f"Model: {model_info['name']}")
    print("-" * 50)

    dataset = datasets[task]

    train_dataset = PLMDataset(model_info, dataset, seqid_dict)

    training_args = TrainingArguments(
        output_dir=settings.FINETUNED_RESULTS_PATH,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        do_train=True,
        do_eval=False,
        evaluation_strategy="no",
        gradient_accumulation_steps=64,
        logging_dir="./logs",
        logging_steps=200,
        fp16=True,
        seed=settings.SEED,
        warmup_steps=1000,
    )

    plm_model = AutoModelForSequenceClassification.from_pretrained(
        model_info["model"], num_labels=2
    )

    trainer = Trainer(model=plm_model, args=training_args, train_dataset=train_dataset)

    trainer.train()

    # Create the output directory if it doesn't exist
    output_dir = f"{settings.FINETUNED_MODELS_PATH}/{model_info['name']}/{task}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the fine-tuned model
    trainer.save_model(output_dir)
