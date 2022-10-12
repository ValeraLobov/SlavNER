import os
import gc
import re
from os.path import dirname

import numpy as np
import random
import datasets
import torch
import torch.optim as optim
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification, \
    XLMRobertaForTokenClassification, AutoModelForTokenClassification
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from huawei_slavic_ner_models.src.utils import config, tokenizer, logging, device, tag_to_ix, compute_similarity
from huawei_slavic_ner_models.src.evaluate import compute_metrics
from huawei_slavic_ner_models.data.preprocess import tokenize_and_align_labels
from huawei_slavic_ner_models.data.dataset import (
    create_huggingface_dataset, create_synthetic_dataset_s1, create_synthetic_dataset_s2, create_synthetic_dataset_s3,
    get_s1_data, get_s2_data, get_s3_data, get_wikiner_dataset, get_slavicner_dataset, get_synth_injection
)
from huawei_slavic_ner_models.src.train_cl_model import BiLSTM_CRF, train_cl_model, prepare_sequence

BASEDIR = dirname(dirname(os.path.abspath(__file__)))


class TrainerWithoutShuffle(Trainer):
    """
    Huggingface Trainer class with sequential sampling (needed for curriculum learning, default is random sampling)
    """
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=SequentialSampler(train_dataset),
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def ner_train_model(model, train_dataset_tokenized, val_dataset_tokenized, shuffle=True, num_epochs=config.backbone_model.num_epochs):
    """
    main function for training XLM-R model
    :param model: XLMRobertaForTokenClassification input model
    :param train_dataset_tokenized: preprocessed Dataset[]
    :param val_dataset_tokenized: preprocessed Dataset[]
    :param shuffle: bool - shuffle train dataset or not
    :param num_epochs: number of epochs for training
    :return: model: XLMRobertaForTokenClassification, metrics: Dict[str]
    """
    model_name = f"{config.ner_task.model_checkpoint_name}_{random.getrandbits(16)}"
    args = TrainingArguments(
        model_name,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.backbone_model.learning_rate,
        per_device_train_batch_size=config.backbone_model.batch_size,
        per_device_eval_batch_size=config.backbone_model.batch_size,
        num_train_epochs=config.backbone_model.num_epochs,
        weight_decay=config.backbone_model.weight_decay,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model='overall_f1'
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    if shuffle:
        trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset_tokenized,
            eval_dataset=val_dataset_tokenized,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
    else:
        trainer = TrainerWithoutShuffle(
            model,
            args,
            train_dataset=train_dataset_tokenized,
            eval_dataset=val_dataset_tokenized,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

    trainer.train()
    metrics = trainer.evaluate()
    return model, metrics


def rate_by_sentence_length(train_dataset, val_dataset, logging, wechsel_model_path, source_language_vocab_token_inds,
                            num_epochs=config.backbone_model.num_epochs):
    """
    Model C1: Order train dataset by sentence length, train XLM-R and compute metrics
    :param train_dataset: preprocessed Dataset[]
    :param val_dataset: preprocessed Dataset[]
    :param logging: logging() object
    :param wechsel_model_path: str - path to wechsel model, which will be updated on every iteration (loaded, fine-tuned, saved)
    :param source_language_vocab_token_inds: set of vocab token inds, which is used as WECHSEL base vocabulary
    :param num_epochs: number of epochs for training
    """
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
    else:
        raise Exception("Model backbone is undefined. Please use 'xlm' or 'rubert' in config")
    if wechsel_model_path:
        wechsel_model.load_state_dict(torch.load(wechsel_model_path, map_location='cuda'))
        wechsel_model.eval()
        wechsel_model = wechsel_model_init(wechsel_model, train_dataset, tokenizer, source_language_vocab_token_inds)

    sentence_length_indicies = sorted(range(len(train_dataset)), key=lambda x: len(train_dataset[x]['tokens']))
    train_dataset_sorted_by_sentence_length = train_dataset.select(sentence_length_indicies)

    wechsel_model_tuned, metrics = ner_train_model(wechsel_model, train_dataset_sorted_by_sentence_length, val_dataset,
                                                   False, num_epochs)

    logging.info("Sentence length(C1) model")
    logging.info(f"Metrics: {metrics}")

    # save model for next iteration
    if wechsel_model_path:
        torch.save(wechsel_model_tuned.state_dict(), wechsel_model_path)
    else:
        torch.save(wechsel_model_tuned.state_dict(), os.path.join(BASEDIR, config.backbone_model.wechsel_c1_model_path))

    del wechsel_model, wechsel_model_tuned
    gc.collect()
    torch.cuda.empty_cache()


def rate_by_entities_prob(train_dataset, val_dataset, train_texts, train_tags, logging, wechsel_model_path,
                          source_language_vocab_token_inds, num_epochs=config.backbone_model.num_epochs):
    """
    Model C2: Order train dataset by entities prob from CL model, train XLM-R and compute metrics
    :param train_dataset: preprocessed Dataset[]
    :param val_dataset: preprocessed Dataset[]
    :param train_texts: List[List[str]]
    :param train_tags: List[List[str]]
    :param logging: logging() object
    :param wechsel_model_path: str - path to wechsel model, which will be updated on every iteration (loaded, fine-tuned, saved)
    :param source_language_vocab_token_inds: set of vocab token inds, which is used as WECHSEL base vocabulary
    :param num_epochs: number of epochs for training
    :return: CL model for perplexity (C3) model
    """
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
    else:
        raise Exception("Model backbone is undefined. Please use 'xlm' or 'rubert' in config")
    if wechsel_model_path:
        wechsel_model.load_state_dict(torch.load(wechsel_model_path, map_location='cuda'))
        wechsel_model.eval()
        wechsel_model = wechsel_model_init(wechsel_model, train_dataset, tokenizer, source_language_vocab_token_inds)

    samples_stats = {}
    word_to_ix = {}
    for sentence, tags in zip(train_texts, train_tags):
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    logging.info("Entities prob(C2) model")

    model = BiLSTM_CRF(
        len(word_to_ix),
        tag_to_ix,
        config.curriculum_learning_model.EMBEDDING_DIM,
        config.curriculum_learning_model.HIDDEN_DIM
    ).to('cpu')
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.curriculum_learning_model.LR,
        weight_decay=config.curriculum_learning_model.WEIGHT_DECAY
    )
    model = train_cl_model(model, optimizer, train_texts, train_tags, word_to_ix)

    for ind, (sentence, tags) in tqdm(enumerate(zip(train_texts, train_tags))):
        sentence_in = prepare_sequence(sentence, word_to_ix).to('cpu')
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long).to('cpu')
        feats = model._get_lstm_features(sentence_in)

        entities_feats = torch.index_select(feats, 0, targets.nonzero(as_tuple=True)[0])

        if len(entities_feats) == 0:
            samples_stats[ind] = {}
            samples_stats[ind]['entities_prob'] = 0
            continue

        entities_mean_prob = torch.index_select(entities_feats, 1, targets[targets > 0][0]).mean()
        samples_stats[ind] = {}
        samples_stats[ind]['entities_prob'] = entities_mean_prob.item()

    entities_prob_indicies = sorted(list(samples_stats.keys()), key=lambda x: samples_stats[x]['entities_prob'],
                                    reverse=True)
    mixed_train_dataset_sorted_by_entities_prob = train_dataset.select(entities_prob_indicies)

    wechsel_model_tuned, metrics = ner_train_model(wechsel_model, mixed_train_dataset_sorted_by_entities_prob, val_dataset,
                                          False, num_epochs)
    logging.info(f"Metrics: {metrics}")

    # save model for next iteration
    if wechsel_model_path:
        torch.save(wechsel_model_tuned.state_dict(), wechsel_model_path)
    else:
        torch.save(wechsel_model_tuned.state_dict(), os.path.join(BASEDIR, config.backbone_model.wechsel_c2_model_path))

    del wechsel_model, wechsel_model_tuned
    gc.collect()
    torch.cuda.empty_cache()
    return model


def rate_by_perplexity(train_dataset, val_dataset, train_texts, train_tags, cl_model, logging, wechsel_model_path,
                       source_language_vocab_token_inds, num_epochs=config.backbone_model.num_epochs):
    """
    Model C3: Order train dataset by perplexity from CL model, train XLM-R and compute metrics
    :param train_dataset: preprocessed Dataset[]
    :param val_dataset: preprocessed Dataset[]
    :param train_texts: List[List[str]]
    :param train_tags: List[List[str]]
    :param cl_model: BiLSTM-CRF model, which was trained in C2 step
    :param logging: logging() object
    :param wechsel_model_path: str - path to wechsel model, which will be updated on every iteration (loaded, fine-tuned, saved)
    :param source_language_vocab_token_inds: set of vocab token inds, which is used as WECHSEL base vocabulary
    :param num_epochs: number of epochs for training
    """

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
    else:
        raise Exception("Model backbone is undefined. Please use 'xlm' or 'rubert' in config")
    if wechsel_model_path:
        wechsel_model.load_state_dict(torch.load(wechsel_model_path, map_location='cuda'))
        wechsel_model.eval()
        wechsel_model = wechsel_model_init(wechsel_model, train_dataset, tokenizer, source_language_vocab_token_inds)

    training_data = zip(train_texts, train_tags)
    word_to_ix = {}

    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    logging.info("Perplexity(C3) model")

    peplexities = []
    for sentence, tags in tqdm(zip(train_texts, train_tags)):
        sentence_in = prepare_sequence(sentence, word_to_ix).to('cpu')
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long).to('cpu')

        perplexity = 2 ** (cl_model.neg_log_likelihood(sentence_in, targets))
        peplexities.append(perplexity.item())

    perplexity_indicies = np.argsort(peplexities)[::-1]

    train_dataset_sorted_by_perplexity = train_dataset.select(perplexity_indicies)

    wechsel_model_tuned, metrics = ner_train_model(wechsel_model, train_dataset_sorted_by_perplexity, val_dataset, shuffle=False,
                                              num_epochs=num_epochs)
    logging.info(f"Metrics: {metrics}")
    # save model for next iteration
    if wechsel_model_path:
        torch.save(wechsel_model_tuned.state_dict(), wechsel_model_path)
    else:
        torch.save(wechsel_model_tuned.state_dict(), os.path.join(BASEDIR, config.backbone_model.wechsel_c3_model_path))

    del wechsel_model, wechsel_model_tuned
    gc.collect()
    torch.cuda.empty_cache()


def wechsel_model_init(model_base, train_dataset_tokenized, tokenizer, source_language_vocab_token_inds):
    """
    WECHSEL algorithm: replace vocab token embeddings of model_base in target language with top5 nearest token embeddings in
    source language.
    :param model_base: XLMRobertaForTokenClassification, which will be used for WECHSEL
    :param train_dataset_tokenized: preprocessed Dataset[]
    :param tokenizer: XLMRobertaTokenizerFast
    :param source_language_vocab_token_inds: set of vocab token inds, which is used as WECHSEL base vocabulary
    :return: XLMRobertaForTokenClassification model after WECHSEL transformation of embeddings
    """
    initial_vocab_embeds = model_base.get_input_embeddings().state_dict()['weight'].cpu().numpy()

    target_vocab_token_inds = set()
    for data in train_dataset_tokenized:
        for ind in data['input_ids']:
            text_token = tokenizer.convert_ids_to_tokens([ind])[0]
            if re.search(r'\d', text_token):
                continue
            target_vocab_token_inds.add(ind)
    logging.info(f"vocab size: {len(target_vocab_token_inds)}")

    val_tokens_similarities = np.dot(initial_vocab_embeds[list(target_vocab_token_inds), :], initial_vocab_embeds[list(source_language_vocab_token_inds), :].T)

    NEIGHBORS_NUM = config.wechsel_model.num_neighbors

    val_similar_tokens = {}
    val_similar_tokens_softmax_scores = {}

    target_vocab_token_inds = list(target_vocab_token_inds)
    source_language_vocab_token_inds = list(source_language_vocab_token_inds)

    for ind, row in tqdm(enumerate(val_tokens_similarities)):
        val_similar_tokens_softmax_scores[target_vocab_token_inds[ind]] = softmax(sorted(row, reverse=True)[:NEIGHBORS_NUM])
        val_similar_tokens[target_vocab_token_inds[ind]] = [source_language_vocab_token_inds[tok] for tok in np.argsort(row)[::-1][:NEIGHBORS_NUM]]

    wechsel_embeddings = {}
    for token_ind in val_similar_tokens:
        closest_embeds = np.array([initial_vocab_embeds[tok] for tok in val_similar_tokens[token_ind]])
        softmaxed_embeds = [closest_embeds[i]*val_similar_tokens_softmax_scores[token_ind][i] for i in range(NEIGHBORS_NUM)]
        wechsel_embeddings[token_ind] = np.sum(softmaxed_embeds, axis=0)

    embeds_cosine_difference = []
    for token_ind in wechsel_embeddings:
        embeds_cosine_difference.append(compute_similarity(wechsel_embeddings[token_ind], initial_vocab_embeds[token_ind]))

    for ind in wechsel_embeddings:
        initial_vocab_embeds[ind] = wechsel_embeddings[ind]

    new_model_input_embeds = model_base.get_input_embeddings()
    new_model_input_embeds.weight.data.copy_(torch.from_numpy(initial_vocab_embeds))
    model_base.set_input_embeddings(new_model_input_embeds)

    return model_base


def train_wechsel_on_s1(langs):
    """
    Train WECHSEL model on dataset S1
    :param langs: List[str] of used languages to train WECHSEL
    """
    wikiner = get_wikiner_dataset()
    locs_names, locs_sentences, orgs_names, orgs_sentences, pers_names, pers_sentences = get_s1_data()

    for lang_index, lang in tqdm(enumerate(langs)):
        logging.info(f"Current language: {lang}")
        generated_locs_sentences, generated_locs_tags = create_synthetic_dataset_s1(
            locs_sentences[lang], locs_names[lang], '[loc]', 'LOC', num_samples_per_sentence=4
        )

        generated_orgs_sentences, generated_orgs_tags = create_synthetic_dataset_s1(
            orgs_sentences[lang], orgs_names[lang], '[???]', 'ORG', num_samples_per_sentence=13
        )

        generated_pers_sentences, generated_pers_tags = create_synthetic_dataset_s1(
            pers_sentences[lang], pers_names[lang], '[male]', 'PER', num_samples_per_sentence=1
        )

        synthetic_dataset_sentences = generated_locs_sentences[:config.ner_task.loc_training_samples] + \
                                      generated_orgs_sentences[:config.ner_task.org_training_samples] + \
                                      generated_pers_sentences[:config.ner_task.per_male_training_samples + config.ner_task.per_female_training_samples]
        synthetic_dataset_tags = generated_locs_tags[:config.ner_task.loc_training_samples] + \
                                 generated_orgs_tags[:config.ner_task.org_training_samples] + \
                                 generated_pers_tags[:config.ner_task.per_male_training_samples + config.ner_task.per_female_training_samples]

        synth_train_texts, synth_val_texts, synth_train_tags, synth_val_tags = train_test_split(
            synthetic_dataset_sentences,
            synthetic_dataset_tags,
            test_size=.2
        )

        logging.info(f"Dataset S1, train/val samples num: {(len(synth_train_texts), len(synth_val_texts))}")

        synth_train_dataset = create_huggingface_dataset(synth_train_texts, synth_train_tags)
        synth_train_dataset_tokenized = synth_train_dataset.map(tokenize_and_align_labels, batched=True)

        if lang_index == 0:
            # first language is source language for WECHSEL
            source_language_vocab_token_inds = set()
            for data in synth_train_dataset_tokenized:
                for ind in data['input_ids']:
                    text_token = tokenizer.convert_ids_to_tokens([ind])[0]
                    if re.search(r'\d', text_token):
                        continue
                    source_language_vocab_token_inds.add(ind)
            sentence_len_wechsel_path, entities_prob_wechsel_path, perplexity_wechsel_path = '', '', ''
        else:
            sentence_len_wechsel_path = os.path.join(BASEDIR, config.backbone_model.wechsel_c1_model_path)
            entities_prob_wechsel_path = os.path.join(BASEDIR, config.backbone_model.wechsel_c2_model_path)
            perplexity_wechsel_path = os.path.join(BASEDIR, config.backbone_model.wechsel_c3_model_path)

        # validation dataset: WikiNER
        wikiner_val_dataset = create_huggingface_dataset(wikiner[lang]['texts'], wikiner[lang]['tags'])
        wikiner_val_dataset_tokenized = wikiner_val_dataset.map(tokenize_and_align_labels, batched=True)

        rate_by_sentence_length(synth_train_dataset_tokenized, wikiner_val_dataset_tokenized, logging,
                                sentence_len_wechsel_path, source_language_vocab_token_inds)
        cl_model = rate_by_entities_prob(synth_train_dataset_tokenized, wikiner_val_dataset_tokenized,
                                         synth_train_texts, synth_train_tags, logging, entities_prob_wechsel_path, source_language_vocab_token_inds)
        rate_by_perplexity(synth_train_dataset_tokenized, wikiner_val_dataset_tokenized, synth_train_texts,
                           synth_train_tags, cl_model, logging, perplexity_wechsel_path, source_language_vocab_token_inds)


def train_wechsel_on_s2(langs):
    """
    Train WECHSEL model on dataset S2
    :param langs: List[str] of used languages to train WECHSEL
    """
    wikiner = get_wikiner_dataset()
    locs_names, locs_sentences, orgs_names, orgs_sentences, male_pers_names, female_pers_names, pers_sentences = get_s2_data()

    for lang_index, lang in tqdm(enumerate(langs)):
        logging.info(f"Current language: {lang}")

        generated_locs_sentences, generated_locs_tags = create_synthetic_dataset_s2(
            locs_sentences[lang], locs_names[lang], '[???]', 'LOC', num_samples_per_sentence=3
        )

        generated_orgs_sentences, generated_orgs_tags = create_synthetic_dataset_s2(
            orgs_sentences[lang], orgs_names[lang], '[???]', 'ORG', num_samples_per_sentence=12
        )

        generated_male_pers_sentences, generated_male_pers_tags = create_synthetic_dataset_s2(
            pers_sentences[lang], male_pers_names[lang], '[male]', 'PER', num_samples_per_sentence=1
        )

        generated_female_pers_sentences, generated_female_pers_tags = create_synthetic_dataset_s2(
            pers_sentences[lang], female_pers_names[lang], '[female]', 'PER', num_samples_per_sentence=4
        )

        synthetic_dataset_sentences = generated_locs_sentences[:config.ner_task.loc_training_samples] + \
                                      generated_orgs_sentences[:config.ner_task.org_training_samples] + \
                                      generated_male_pers_sentences[:config.ner_task.per_male_training_samples] + \
                                      generated_female_pers_sentences[:config.ner_task.per_female_training_samples]
        synthetic_dataset_tags = generated_locs_tags[:config.ner_task.loc_training_samples] + \
                                 generated_orgs_tags[:config.ner_task.org_training_samples] + \
                                 generated_male_pers_tags[:config.ner_task.per_male_training_samples] + \
                                 generated_female_pers_tags[:config.ner_task.per_female_training_samples]

        synth_train_texts, synth_val_texts, synth_train_tags, synth_val_tags = train_test_split(
            synthetic_dataset_sentences,
            synthetic_dataset_tags,
            test_size=.2
        )

        logging.info(f"Dataset S2, train/val samples num: {(len(synth_train_texts), len(synth_val_texts))}")

        synth_train_dataset = create_huggingface_dataset(synth_train_texts, synth_train_tags)
        synth_train_dataset_tokenized = synth_train_dataset.map(tokenize_and_align_labels, batched=True)

        if lang_index == 0:
            # first language is source language for WECHSEL
            source_language_vocab_token_inds = set()
            for data in synth_train_dataset_tokenized:
                for ind in data['input_ids']:
                    text_token = tokenizer.convert_ids_to_tokens([ind])[0]
                    if re.search(r'\d', text_token):
                        continue
                    source_language_vocab_token_inds.add(ind)
            sentence_len_wechsel_path, entities_prob_wechsel_path, perplexity_wechsel_path = '', '', ''
        else:
            sentence_len_wechsel_path = os.path.join(BASEDIR, config.backbone_model.wechsel_c1_model_path)
            entities_prob_wechsel_path = os.path.join(BASEDIR, config.backbone_model.wechsel_c2_model_path)
            perplexity_wechsel_path = os.path.join(BASEDIR, config.backbone_model.wechsel_c3_model_path)

        # validation dataset: WikiNER
        wikiner_val_dataset = create_huggingface_dataset(wikiner[lang]['texts'], wikiner[lang]['tags'])
        wikiner_val_dataset_tokenized = wikiner_val_dataset.map(tokenize_and_align_labels, batched=True)

        rate_by_sentence_length(synth_train_dataset_tokenized, wikiner_val_dataset_tokenized, logging,
                                sentence_len_wechsel_path, source_language_vocab_token_inds)
        cl_model = rate_by_entities_prob(synth_train_dataset_tokenized, wikiner_val_dataset_tokenized,
                                         synth_train_texts, synth_train_tags, logging, entities_prob_wechsel_path, source_language_vocab_token_inds)
        rate_by_perplexity(synth_train_dataset_tokenized, wikiner_val_dataset_tokenized, synth_train_texts,
                           synth_train_tags, cl_model, logging, perplexity_wechsel_path, source_language_vocab_token_inds)


# TODO: combine S2 and S3 in single function
def train_wechsel_on_s3(langs):
    """
    Train WECHSEL model on dataset S3
    :param langs: List[str] of used languages to train WECHSEL
    """
    wikiner = get_wikiner_dataset()
    locs_names, locs_sentences, orgs_names_dict, orgs_sentences, male_pers_names, female_pers_names, pers_sentences = get_s3_data()

    for lang_index, lang in tqdm(enumerate(langs)):
        logging.info(f"Current language: {lang}")

        generated_locs_sentences, generated_locs_tags = create_synthetic_dataset_s3(
            locs_sentences[lang], {'[???]': locs_names[lang]}, ['[???]'], 'LOC', num_samples_per_sentence=3
        )

        generated_orgs_sentences, generated_orgs_tags = create_synthetic_dataset_s3(
            orgs_sentences[lang], orgs_names_dict[lang], ['[n]', '[m]', '[f]', '[pl]'], 'ORG',
            num_samples_per_sentence=5
        )

        generated_male_pers_sentences, generated_male_pers_tags = create_synthetic_dataset_s3(
            pers_sentences[lang], male_pers_names[lang], ['[male]'], 'PER', num_samples_per_sentence=1
        )

        generated_female_pers_sentences, generated_female_pers_tags = create_synthetic_dataset_s3(
            pers_sentences[lang], female_pers_names[lang], ['[female]'], 'PER', num_samples_per_sentence=5
        )

        synthetic_dataset_sentences = generated_locs_sentences[:config.ner_task.loc_training_samples] + \
                                      generated_orgs_sentences[:config.ner_task.org_training_samples] + \
                                      generated_male_pers_sentences[:config.ner_task.per_male_training_samples] + \
                                      generated_female_pers_sentences[:config.ner_task.per_female_training_samples]
        synthetic_dataset_tags = generated_locs_tags[:config.ner_task.loc_training_samples] + \
                                 generated_orgs_tags[:config.ner_task.org_training_samples] + \
                                 generated_male_pers_tags[:config.ner_task.per_male_training_samples] + \
                                 generated_female_pers_tags[:config.ner_task.per_female_training_samples]

        synth_train_texts, synth_val_texts, synth_train_tags, synth_val_tags = train_test_split(
            synthetic_dataset_sentences,
            synthetic_dataset_tags,
            test_size=.2
        )

        logging.info(f"Dataset S3, train/val samples num: {(len(synth_train_texts), len(synth_val_texts))}")

        synth_train_dataset = create_huggingface_dataset(synth_train_texts, synth_train_tags)
        synth_train_dataset_tokenized = synth_train_dataset.map(tokenize_and_align_labels, batched=True)

        if lang_index == 0:
            # first language is source language for WECHSEL
            source_language_vocab_token_inds = set()
            for data in synth_train_dataset_tokenized:
                for ind in data['input_ids']:
                    text_token = tokenizer.convert_ids_to_tokens([ind])[0]
                    if re.search(r'\d', text_token):
                        continue
                    source_language_vocab_token_inds.add(ind)
            sentence_len_wechsel_path, entities_prob_wechsel_path, perplexity_wechsel_path = '', '', ''
        else:
            sentence_len_wechsel_path = os.path.join(BASEDIR, config.backbone_model.wechsel_c1_model_path)
            entities_prob_wechsel_path = os.path.join(BASEDIR, config.backbone_model.wechsel_c2_model_path)
            perplexity_wechsel_path = os.path.join(BASEDIR, config.backbone_model.wechsel_c3_model_path)

        # validation dataset: WikiNER
        wikiner_val_dataset = create_huggingface_dataset(wikiner[lang]['texts'], wikiner[lang]['tags'])
        wikiner_val_dataset_tokenized = wikiner_val_dataset.map(tokenize_and_align_labels, batched=True)

        rate_by_sentence_length(synth_train_dataset_tokenized, wikiner_val_dataset_tokenized, logging,
                                sentence_len_wechsel_path, source_language_vocab_token_inds)
        cl_model = rate_by_entities_prob(synth_train_dataset_tokenized, wikiner_val_dataset_tokenized,
                                         synth_train_texts, synth_train_tags, logging, entities_prob_wechsel_path, source_language_vocab_token_inds)
        rate_by_perplexity(synth_train_dataset_tokenized, wikiner_val_dataset_tokenized, synth_train_texts,
                           synth_train_tags, cl_model, logging, perplexity_wechsel_path, source_language_vocab_token_inds)


def train_wechsel_on_slavicner(langs):
    """
    Train WECHSEL model on dataset SlavicNER
    :param langs: List[str] of used languages to train WECHSEL
    """

    for lang_index, lang in tqdm(enumerate(langs)):
        logging.info(f"Current language: {lang}")

        train_sentences, train_tags = get_slavicner_dataset(lang, 'train')
        test_sentences, test_tags = get_slavicner_dataset(lang, 'test')
        synth_sentences, synth_tags = get_synth_injection(lang)

        logging.info(f"Dataset SlavicNER, train/val samples num: {(len(train_sentences), len(test_sentences))}")

        synth_train_texts = train_sentences + synth_sentences
        synth_train_tags = train_tags + synth_tags
        synth_train_dataset = create_huggingface_dataset(synth_train_texts, synth_train_tags)
        synth_train_dataset_tokenized = synth_train_dataset.map(tokenize_and_align_labels, batched=True)

        # validation dataset: SlavicNER
        val_dataset = create_huggingface_dataset(test_sentences, test_tags)
        val_dataset_tokenized = val_dataset.map(tokenize_and_align_labels, batched=True)

        if lang_index == 0:
            # first language is source language for WECHSEL
            source_language_vocab_token_inds = set()
            for data in synth_train_dataset_tokenized:
                for ind in data['input_ids']:
                    text_token = tokenizer.convert_ids_to_tokens([ind])[0]
                    if re.search(r'\d', text_token):
                        continue
                    source_language_vocab_token_inds.add(ind)
            sentence_len_wechsel_path, entities_prob_wechsel_path, perplexity_wechsel_path = '', '', ''
        else:
            sentence_len_wechsel_path = os.path.join(BASEDIR, config.backbone_model.wechsel_c1_model_path)
            entities_prob_wechsel_path = os.path.join(BASEDIR, config.backbone_model.wechsel_c2_model_path)
            perplexity_wechsel_path = os.path.join(BASEDIR, config.backbone_model.wechsel_c3_model_path)

        rate_by_sentence_length(synth_train_dataset_tokenized, val_dataset_tokenized, logging,
                                sentence_len_wechsel_path, source_language_vocab_token_inds)
        cl_model = rate_by_entities_prob(synth_train_dataset_tokenized, val_dataset_tokenized,
                                         synth_train_texts, synth_train_tags, logging, entities_prob_wechsel_path, source_language_vocab_token_inds)
        rate_by_perplexity(synth_train_dataset_tokenized, val_dataset_tokenized, synth_train_texts,
                           synth_train_tags, cl_model, logging, perplexity_wechsel_path, source_language_vocab_token_inds)