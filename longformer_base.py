import sys
import json
import random
import os
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
import torch
from transformers import AutoTokenizer, LongformerForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import evaluate
import argparse
import logging

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

pretrain_model = "allenai/longformer-base-4096"
tokenizer = AutoTokenizer.from_pretrained(pretrain_model)

def tokenize_function_1(example):
    return tokenizer(".".join(example["text"]), truncation=True, padding="max_length")

def tokenize_function_2(example):
    return tokenizer(".".join(example["law"]), truncation=True, padding="max_length")

def one_hot_labels(example):
    example["labels"] = [1.0 if l in example["labels"] else 0.0 for l in list(range(10))]
    return example

def train_model(model, train_dataloader, epoch_n):
    logger = logging.getLogger(f"train epoch {epoch_n}")
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_dataloader, desc=f"train epoch {epoch_n}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        criterion = torch.nn.BCELoss(weight=class_weights)
        sigmoid = torch.nn.Sigmoid()
        loss = criterion(sigmoid(logits), batch['labels'])
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        model.zero_grad()
    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"avg_train_loss: {avg_train_loss}")
    logger.info(f"avg_train_loss: {avg_train_loss}")

def cal_test_score(model, test_dataloader, epoch_n):
    logger = logging.getLogger(f"test epoch {epoch_n}")
    metric = []
    for i in range(10):
        metric.append(evaluate.combine(["accuracy", "recall", "precision", "f1"]))
    metric_micro = evaluate.combine(["accuracy", "recall", "precision", "f1"])
    model.eval()
    total_test_loss = 0
    for batch in tqdm(test_dataloader, desc=f"test epoch {epoch_n}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        sigmoid = torch.nn.Sigmoid()
        criterion = torch.nn.BCELoss(weight=class_weights)
        loss = criterion(sigmoid(logits), batch['labels'])
        total_test_loss += loss.item()
        probs = sigmoid(torch.Tensor(logits))
        predictions = [[1 if t > 0.5 else 0 for t in p] for p in probs]
        for i in range(10):
            p = [t[i] for t in predictions]
            r = [t[i] for t in batch["labels"]]
            metric[i].add_batch(predictions=p, references=r)
            metric_micro.add_batch(predictions=p, references=r)
    avg_test_loss = total_test_loss / len(test_dataloader)
    print(f"avg_test_loss: {avg_test_loss}")
    logger.info(f"avg_test_loss: {avg_test_loss}")
    f1_avg = 0
    for idx, m in enumerate(metric):
        scores = m.compute()
        print(f"label {idx}: {str(scores)}")
        logger.info(f"label {idx}: {str(scores)}")
        f1_avg += scores["f1"]
    print(f"f1 macro: {f1_avg/10}")
    logger.info(f"f1 macro: {f1_avg/10}")
    metric_micro_score = metric_micro.compute()
    print(f"f1 micro: {str(metric_micro_score)}")
    logger.info(f"f1 micro: {str(metric_micro_score)}")
    return avg_test_loss

def cal_val_score(model, validation_dataloader, epoch_n):
    logger = logging.getLogger(f"validation epoch {epoch_n}")
    metric = []
    for i in range(10):
        metric.append(evaluate.combine(["accuracy", "recall", "precision", "f1"]))
    metric_micro = evaluate.combine(["accuracy", "recall", "precision", "f1"])
    model.eval()
    total_val_loss = 0
    for batch in tqdm(validation_dataloader, desc=f"val epoch {epoch_n}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        sigmoid = torch.nn.Sigmoid()
        criterion = torch.nn.BCELoss(weight=class_weights)
        loss = criterion(sigmoid(logits), batch['labels'])
        total_val_loss += loss.item()
        probs = sigmoid(torch.Tensor(logits))
        predictions = [[1 if t > 0.5 else 0 for t in p] for p in probs]
        for i in range(10):
            p = [t[i] for t in predictions]
            r = [t[i] for t in batch["labels"]]
            metric[i].add_batch(predictions=p, references=r)
            metric_micro.add_batch(predictions=p, references=r)
    avg_val_loss = total_val_loss / len(validation_dataloader)
    print("avg_val_loss", avg_val_loss)
    logger.info(f"avg_val_loss: {avg_val_loss}")
    f1_avg = 0
    for idx, m in enumerate(metric):
        scores = m.compute()
        print(f"label {idx}: {str(scores)}")
        logger.info(f"label {idx}: {str(scores)}")
        f1_avg += scores["f1"]
    print(f"f1 macro: {f1_avg/10}")
    logger.info(f"f1 macro: {f1_avg/10}")
    metric_micro_score = metric_micro.compute()
    print(f"f1 micro: {str(metric_micro_score)}")
    logger.info(f"f1 micro: {str(metric_micro_score)}")
    return avg_val_loss


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-t", "--task_name", choices=["base", "add_true_label", "add_false_label"], required=True)
        parser.add_argument("--dataset", choices=["ecthr_a", "ecthr_b"], required=True)
        parser.add_argument("-n", "--num_epochs", type=int, default=7)
        parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5)
        parser.add_argument("-p", "--model_saving_path", required=True)
        parser.add_argument("-s", "--seed_number", type=int, default=42)
        parser.add_argument("--test", action="store_true", default=False, help="Run with sampled data for testing")
        parser.add_argument("--log_file", required=True)
        parser.add_argument("--train_batch_size", type=int, default=1)
        parser.add_argument("--eval_batch_size", type=int, default=5)
        
        args = parser.parse_args()

        logging.basicConfig(
            handlers=[
                logging.FileHandler(args.log_file, mode="w"),
                logging.StreamHandler()
            ],
            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S',
            level=logging.DEBUG)

        logging.info(f"Args: {str(args)}")
        task_name = args.task_name
        num_epochs = args.num_epochs
        learning_rate = args.learning_rate
        train_batch_size = args.train_batch_size
        eval_batch_size = args.eval_batch_size
        model_saving_path = args.model_saving_path

        isExist = os.path.exists(model_saving_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(model_saving_path)
            
        seed_number = args.seed_number

        seed_everything(seed_number)

        dataset = load_dataset("huynguyendayrui/ecthr")
        if args.dataset == "ecthr_a":
            dataset = dataset.rename_column("labels_task_a","labels")
            dataset = dataset.remove_columns(["labels_task_b"])
        else:
            dataset = dataset.rename_column("labels_task_b","labels")
            dataset = dataset.remove_columns(["labels_task_a"])

        if args.test:
            dataset["train"] = dataset["train"].select(range(5))
            dataset["test"] = dataset["test"].select(range(5))
            dataset["validation"] = dataset["validation"].select(range(5))

        load_model = pretrain_model

        model = LongformerForSequenceClassification.from_pretrained(load_model, num_labels=10, problem_type="multi_label_classification")
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        tokenized_datasets_fact = dataset.map(tokenize_function_1)
        tokenized_datasets_fact = tokenized_datasets_fact.remove_columns(["text", "law"])
        tokenized_datasets_fact = tokenized_datasets_fact.map(one_hot_labels)
        tokenized_datasets_fact.set_format("torch")
        tokenized_datasets_fact = tokenized_datasets_fact.map(lambda x : {"float_labels": x["labels"].to(torch.float)})
        tokenized_datasets_fact = tokenized_datasets_fact.remove_columns("labels")
        tokenized_datasets_fact = tokenized_datasets_fact.rename_column("float_labels", "labels")

        if task_name == "add_true_label" or task_name == "add_false_label":
            tokenized_datasets_legal = dataset["train"].map(tokenize_function_2)
            tokenized_datasets_legal = tokenized_datasets_legal.remove_columns(["text", "law"])

            if task_name == "add_false_label":
                single_col = "labels" # specify the column name here
                dset_single_col = tokenized_datasets_legal.remove_columns([col for col in tokenized_datasets_legal.column_names if col != single_col])
                dset_single_col_shuffled = dset_single_col.shuffle(seed=seed_number)
                dset_without_single_col = tokenized_datasets_legal.remove_columns([single_col])
                tokenized_datasets_legal = concatenate_datasets([dset_without_single_col, dset_single_col_shuffled], axis = 1)

            tokenized_datasets_legal = tokenized_datasets_legal.map(one_hot_labels)
            tokenized_datasets_legal.set_format("torch")
            tokenized_datasets_legal = tokenized_datasets_legal.map(lambda x : {"float_labels": x["labels"].to(torch.float)})
            tokenized_datasets_legal = tokenized_datasets_legal.remove_columns("labels")
            tokenized_datasets_legal = tokenized_datasets_legal.rename_column("float_labels", "labels")

        if task_name == "base":
            train_dataset = tokenized_datasets_fact["train"]
        else:
            train_dataset = concatenate_datasets([tokenized_datasets_fact["train"], tokenized_datasets_legal])

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)

        test_dataloader = DataLoader(tokenized_datasets_fact["test"], batch_size=eval_batch_size)

        validation_dataloader = DataLoader(tokenized_datasets_fact["validation"], batch_size=eval_batch_size)

        optimizer = AdamW(model.parameters(), lr=learning_rate)

        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )
        if args.dataset == "ecthr_a":
            class_weights = torch.Tensor([
                0.9525375939849624, 
                0.8732142857142857, 
                0.8714285714285714, 
                0.5578947368421052, 
                0.9332706766917294, 
                0.9961466165413534, 
                0.9726503759398496, 
                0.9896616541353384, 
                0.9867481203007519, 
                0.8664473684210526]
            ).to(device)
        else:
            class_weights = torch.Tensor([
                0.9526775541207748,
                0.8678313710596278,
                0.8767185719711356,
                0.5870110140524117,
                0.919787314849981,
                0.9938473224458793,
                0.9665020888720092,
                0.9876946448917584,
                0.9662742119255602,
                0.8816559058108622]
            ).to(device)
        prev_val_loss = 10

        for epoch in range(num_epochs):
            train_model(model, train_dataloader, epoch)
            model_checkpoint = f"longformer_{task_name}_epoch_{epoch}"
            print(f"Saving {model_checkpoint}")
            model.save_pretrained(model_saving_path + "/" + model_checkpoint)
            test_loss = cal_test_score(model, test_dataloader, epoch)
            val_loss = cal_val_score(model, validation_dataloader, epoch)
            if prev_val_loss < val_loss:
                logging.info(f"Early stopping after epoch {epoch}")
                break
            prev_val_loss = val_loss
    except Exception as e:
        logging.exception(e)