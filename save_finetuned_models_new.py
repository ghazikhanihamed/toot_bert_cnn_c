import os
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from settings import settings
from classes.PLMDataset import PLMDataset

# Define the tasks and corresponding models
tasks = {
    "IC-MP": settings.ESM1B,
    "IT-MP": settings.ESM1B,
    "IC-IT": settings.ESM2
}

# Path to your datasets
path = "./dataset/feb112024/raw/"

# Folders representing each class
folders = ["ionchannels", "iontransporters", "mp"]

# Iterate over tasks and their corresponding models
for task, model_info in tasks.items():
    print(f"Task: {task}")
    print(f"Model: {model_info['name']}")
    print("-" * 50)

    # Determine the classes involved in the current task
    classes = task.split("-")
    
    # Prepare datasets for each class involved in the task
    for class_label in classes:
        train_file = f"{path}{class_label}/{class_label}_train.fasta"
        test_file = f"{path}{class_label}/{class_label}_test.fasta"

        # Load and prepare your datasets here; you'll need to implement PLMDataset
        train_dataset = PLMDataset(model_info, train_file, seqid_dict)  # seqid_dict needs to be defined based on your data
        # Repeat for test_dataset if needed

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=f"{settings.FINETUNED_RESULTS_PATH}/{model_info['name']}/{task}",
            num_train_epochs=5,
            per_device_train_batch_size=1,
            do_train=True,
            do_eval=False,
            logging_dir='./logs',
            logging_steps=200,
            fp16=True,
            seed=settings.SEED,
            warmup_steps=1000
        )

        # Initialize the model for sequence classification
        model = AutoModelForSequenceClassification.from_pretrained(model_info["model"], num_labels=len(classes))

        # Initialize the trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset
            # Include eval_dataset=test_dataset if you prepared it
        )

        # Start training
        trainer.train()

        # Create the output directory if it doesn't exist
        output_dir = f"{settings.FINETUNED_MODELS_PATH}/{model_info['name']}/{task}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the fine-tuned model
        trainer.save_model(output_dir)
