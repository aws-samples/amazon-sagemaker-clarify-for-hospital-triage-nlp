#!/usr/bin/env python3

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from datasets import load_from_disk
import random
import logging
import sys
import argparse
import os
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # https://wandb.ai/jack-morris/david-vs-goliath/reports/Does-Model-Size-Matter-A-Comparison-of-BERT-and-DistilBERT--VmlldzoxMDUxNzU
    # The BERT authors recommend fine-tuning for 4 epochs over the following hyperparameter options:
    # batch sizes: 8, 16, 32, 64, 128
    # learning rates: 3e-4, 1e-4, 5e-5, 3e-5
    # first good one 6 epochs, 8 batch, 16 eval, 3e-5 learning rate
    # final results after last epoch:
    
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=4) # default 32 - change to 16 - then changed to 8
    parser.add_argument("--eval_batch_size", type=int, default=8) # default 64 change to 32 - then changed to 16
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=3e-5)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)
    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    # compute metrics function for binary classification
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        # Want to prioritize recall
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, preds)
        #fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=1)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall, "auc": auc}

    # download model from model hub
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=512)

    # Tokenizer helper function
    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True)

    # Tokenize the dataset
    train_dataset = train_dataset.map(tokenize, batched=True)
    print("======= train dataset ===========")
    print(train_dataset)
    test_dataset = test_dataset.map(tokenize, batched=True)

    # Set format for PyTorch
    train_dataset = train_dataset.rename_column("hospital_expire_flag", "labels")
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    print("======== train dataset after column renames prior to training ========")
    print(train_dataset)
    test_dataset = test_dataset.rename_column("hospital_expire_flag", "labels")
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    # define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        # evaluation_strategy="epoch",
        evaluation_strategy="steps",
        # steps below not necessarily required
        eval_steps=500,
        logging_steps=500,
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    # train model
    trainer.train()

    # evaluate model
    print("========= Performance on the training data set =========")
    eval_result = trainer.evaluate(eval_dataset=train_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Training results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")
            print(f"{key} = {value}\n")            
            
    print("========= Performance on the evaluation data set =========")
    eval_result = trainer.evaluate(eval_dataset=test_dataset)
    
    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")
            print(f"{key} = {value}\n")
    # Saves the model to s3
    trainer.save_model(args.model_dir)