from settings import settings
from classes.PLMDataset import PLMDataset
import os
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification
import json

models = [settings.PROTBERT, settings.PROTBERTBFD,
          settings.ESM1B, settings.ESM2]

# We make a list of datasets. We add all the files in the datasets folder except all_sequences.csv and sequence_ids_dict.jsn and those that has test in their name and not the directories
datasets = [f for f in os.listdir(settings.DATASET_PATH) if os.path.isfile(os.path.join(
    settings.DATASET_PATH, f)) and f != "all_sequences.csv" and f != "sequence_ids_dict.jsn" and "test" not in f]

# We load the dictionary to map the labels which are sequence ids to a number
with open(settings.DATASET_PATH + "sequence_ids_dict.jsn", "r") as f:
    seqid_dict = json.load(f)

for dataset in datasets:
    for model in models:
        print("Dataset: ", dataset)
        print("Model: ", model["name"])
        print("-"*50)

        train_dataset = PLMDataset(model, dataset, seqid_dict)

        training_args = TrainingArguments(
            output_dir=settings.FINETUNED_RESULTS_PATH,
            num_train_epochs=5,
            per_device_train_batch_size=1,
            do_train=True,
            do_eval=False,
            evaluation_strategy="no",
            gradient_accumulation_steps=64,
            logging_dir='./logs',
            logging_steps=200,
            fp16=True,
            seed=settings.SEED,
            warmup_steps=1000
            )
        
        plm_model = AutoModelForSequenceClassification.from_pretrained(model["model"], num_labels=2)

        trainer = Trainer(
            model=plm_model,
            args=training_args,
            train_dataset=train_dataset
        )

        trainer.train()

        # We create a folder for the model if it doesn't exist
        if not os.path.exists(settings.FINETUNED_MODELS_PATH + model["name"] + "_" + dataset[:-4]):
            os.makedirs(settings.FINETUNED_MODELS_PATH + model["name"] + "_" + dataset[:-4])

        # We save the model
        trainer.save_model(settings.FINETUNED_MODELS_PATH + model["name"] + "_" + dataset[:-4] + "/")
