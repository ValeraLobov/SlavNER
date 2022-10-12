import os
from os.path import dirname

import numpy as np
import torch
from tqdm import tqdm
from transformers import XLMRobertaForTokenClassification, AutoModelForTokenClassification, pipeline
from datasets import load_dataset, load_metric
from typing import Tuple

from huawei_slavic_ner_models.src.utils import config, device, logging, tokenizer, id2tag
from huawei_slavic_ner_models.data.preprocess import tokenize_and_align_labels
from huawei_slavic_ner_models.data.dataset import create_huggingface_dataset, get_wikiner_dataset, get_custom_dataset

metric = load_metric("seqeval")


def compute_metrics(p: Tuple):
    """
    compute basic metrics for NER task per entity type
    :param p: Tuple of predictions and labels (used in Huggingface Trainer)
    :return: results: Dict - metrics info
    """
    label_list = ['B-'+item for item in config.ner_task.unique_tags] + ['B-O']
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return results


# TODO: optimize, change to Huggingface Pipeline
def analyze_entities_metrics(val_dataset, input_model=None):
    """
    compute validation metrics for particular model and dataset
    :param val_dataset: preprocessed Dataset[]
    :param input_model: XLMRobertaForTokenClassification()
    :return:
    """

    def compute_metrics_v2(predictions, references):
        tags_set = set()
        tags_set_true_labels = set()
        label_list = ['B-' + item for item in config.ner_task.unique_tags] + ['B-O']
        labels = references
        # predictions = np.argmax(predictions, axis=1)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        tags_set.update(true_predictions[0])
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        tags_set_true_labels.update(true_labels[0])
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return results

    val_predictions = []
    val_references = []
    nb_eval_examples, nb_eval_steps = 0, 0

    for batch_num in tqdm(range(len(val_dataset))):
        input_ids = torch.Tensor([val_dataset[batch_num]['input_ids']]).to(device, dtype=torch.long)
        mask = torch.Tensor([val_dataset[batch_num]['attention_mask']]).to(device, dtype=torch.long)
        labels = torch.Tensor([val_dataset[batch_num]['labels']]).to(device, dtype=torch.long)

        outputs = input_model(input_ids=input_ids, attention_mask=mask, labels=labels, output_hidden_states=True)

        eval_logits = outputs[1]
        hidden_states = outputs[2]

        # compute evaluation accuracy
        flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
        active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)
        labels = torch.masked_select(flattened_targets, active_accuracy)

        nb_eval_steps += 1
        nb_eval_examples += labels.size(0)

        # last logits from whole model
        active_logits = eval_logits.view(-1, input_model.num_labels)  # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        val_predictions.append(predictions.cpu().numpy())
        val_references.append(labels.cpu().numpy())

    return compute_metrics_v2(predictions=val_predictions, references=val_references)


def write_predictions_conll_format(val_dataset, model, filepath):
    """
    Write predictions of model in CoNLL format in same directory
    :param val_dataset: preprocessed Dataset[]
    :param model: XLMRobertaForTokenClassification()
    :param filepath: str
    """
    device_num = 0 if 'cuda' in device.type else -1
    nlp = pipeline('ner', model=model, tokenizer=tokenizer, device=device_num)

    with open(filepath, 'w') as f:
        for data in val_dataset:
            predictions = nlp(' '.join(data['tokens']))
            for entity in predictions:
                tag_id = int(entity['entity'].split('_')[-1])
                f.write(f"{entity['word']}\t{id2tag[tag_id]}\n")
            f.write("\n")


def evaluate_models(model_path: str, dataset_path: str):
    """
    Evaluate model on dataset
    :param model_path: absolute path to XLMRobertaForTokenClassification model file
    :param dataset_path: absolute path to dataset CSV file
    """
    logging.info(f"Evaluate model in path: {model_path}")
    wikiner = get_wikiner_dataset()
    results = {
        lang: [] for lang in config.ner_task.language_order
    }

    if dataset_path:
        logging.info(f"Evaluation on custom dataset in directory data/custom_data, filename: {dataset_path}")
        sentences, tags = get_custom_dataset(dataset_path)
        custom_val_dataset = create_huggingface_dataset(sentences, tags)
        custom_val_dataset_tokenized = custom_val_dataset.map(tokenize_and_align_labels, batched=True)
    else:
        logging.info(f"Evaluation on dataset WikiNER")
        wikiner_val_dataset_tokenized = {}
        for lang in config.ner_task.language_order:
            wikiner_val_dataset = create_huggingface_dataset(wikiner[lang]['texts'], wikiner[lang]['tags'])
            wikiner_val_dataset_tokenized[lang] = wikiner_val_dataset.map(tokenize_and_align_labels, batched=True)

    logging.info(f"Loading model: {model_path}")
    predictions_filepath = os.path.join(dirname(os.path.abspath(model_path)), f"{os.path.basename(model_path)}_predictions.csv")

    if config.backbone_model.base == "xlm":
        wechsel_model = XLMRobertaForTokenClassification.from_pretrained(
            "xlm-roberta-base",
            num_labels=config.ner_task.num_labels
        ).to(device)
    elif config.backbone_model.base == "rubert":
        wechsel_model = AutoModelForTokenClassification.from_pretrained(
            "DeepPavlov/rubert-base-cased",
            num_labels=config.ner_task.num_labels
        ).to(device)
    wechsel_model.load_state_dict(torch.load(model_path, map_location=device))
    wechsel_model.eval()

    if dataset_path:
        metrics = analyze_entities_metrics(custom_val_dataset_tokenized, wechsel_model)
        write_predictions_conll_format(custom_val_dataset_tokenized, wechsel_model, predictions_filepath)
        logging.info(f"Custom validation dataset, metrics: {metrics}")
    else:
        for lang in config.ner_task.language_order:
            metrics = analyze_entities_metrics(wikiner_val_dataset_tokenized[lang], wechsel_model)
            write_predictions_conll_format(wikiner_val_dataset_tokenized[lang], wechsel_model, predictions_filepath)
            results[lang].append(metrics['overall_f1'])
            logging.info(f"WikiNER, lang: {lang}, metrics: {metrics}")
