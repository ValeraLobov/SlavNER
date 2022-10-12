import os
import math
import random
import re
import pickle
import ast
from os.path import dirname
from typing import List, Dict

import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from datasets import Dataset

from huawei_slavic_ner_models.data.preprocess import tokenize_and_align_labels
from huawei_slavic_ner_models.src.utils import config, tag2id

BASEDIR = dirname(os.path.abspath(__file__))


def create_huggingface_dataset(texts: List[List[str]], tags: List[List[str]]):
    """
    dataset processing for huggingface input format
    :param texts List of sentences
    :param tags List of corresponding tags
    :return: Dataset[] for Huggingface Trainer
    """
    dataset = defaultdict(list)

    for ind, text in enumerate(texts):
        dataset['id'].append(str(ind))
        dataset['ner_tags'].append([tag2id[tag] for tag in tags[ind]])
        dataset['tokens'].append([str(item) for item in text])

    dataset = Dataset.from_dict(dataset)
    return dataset


def refactor_tags_to_bio_format(tags: List[List[str]]) -> List[List[str]]:
    """
    Change NER tagging schema to BIO format ('B-TAG' for first tag in entity, 'I-TAG' - for others)
    :param tags: list of tags sequences
    :return: preprocessed tags in BIO format
    """
    for seq_ind, tag_sequence in enumerate(tags):
        new_sequence = []
        for ind, tag in enumerate(tag_sequence):
            tag = tag.split('-')[-1]
            if ind > 0 and tag != 'O' and tag_sequence[ind-1] != 'O':
                new_sequence.append(f'I-{tag}')
            elif ind > 0 and tag != 'O':
                new_sequence.append(f'B-{tag}')
            elif ind == 0 and tag != 'O':
                new_sequence.append(f'B-{tag}')
            else:
                new_sequence.append('O')
        tags[seq_ind] = new_sequence
    return tags


def get_slavicner_dataset(lang: str, mode='train'):
    """
    Read SlavicNER data for particular language
    :param lang: return dataset for selected language
    :param mode: 'train' or 'test', return train or test dataset.
    :return: sentences: List[List[str]], tags: List[List[str]] - SlavicNER dataset
    """
    datasets_dir = os.path.join(BASEDIR, 'SlavicNER')
    used_datasets_paths = []
    banned_datasets = ['covid-19', 'us_election_2020', 'nord_stream', 'ryanair', 'other']

    for dataset_name in os.listdir(datasets_dir):
        if dataset_name not in banned_datasets:
            if lang == 'bg' and (dataset_name == 'ec' or dataset_name == 'trump'):
                continue
            if mode is 'train':
                path_to_data = f'splits/{lang}/train_{lang}.csv'
                used_datasets_paths.append(os.path.join(datasets_dir, dataset_name, path_to_data))
                path_to_data = f'splits/{lang}/dev_{lang}.csv'
                used_datasets_paths.append(os.path.join(datasets_dir, dataset_name, path_to_data))
            else:
                path_to_data = f'splits/{lang}/test_{lang}.csv'
                used_datasets_paths.append(os.path.join(datasets_dir, dataset_name, path_to_data))

    all_sentences = []
    all_tags = []
    replaces_count = 0

    for dataset_path in used_datasets_paths:
        df = pd.read_csv(dataset_path)
        sentences, tags = [], []
        cur_sentence_id = -1
        for ind, row in df.iterrows():
            if row['sentenceId'] == cur_sentence_id:
                sentences.append(row['text'])
                if row['ner'] in config.ner_task.unique_tags:
                    tags.append(row['ner'])
                else:
                    replaces_count += 1
                    tags.append('O')
            else:
                if len(set(tags)) > 1:
                    all_sentences.append(sentences)
                    all_tags.append(tags)
                if row['ner'] not in config.ner_task.unique_tags:
                    row['ner'] = 'O'
                sentences, tags = [row['text']], [row['ner']]
                cur_sentence_id = row['sentenceId']
    return all_sentences, all_tags


def get_synth_injection(lang: str = 'pl'):
    """
    Generate synthetic dataset for SlavicNER training
    :param lang: return dataset for selected language
    :return: sentences: List[List[str]], tags: List[List[str]] - synthetic dataset
    """
    pro_sentences = pd.read_excel(os.path.join(BASEDIR, f'SlavicNER_synth_injection/pro_sents.xlsx'), engine='openpyxl')[lang].values.astype(str)
    pro_entities = pd.read_excel(os.path.join(BASEDIR, f'SlavicNER_synth_injection/pro_names.xlsx'), engine='openpyxl')[lang].values.astype(str)

    evt_sentences = pd.read_excel(os.path.join(BASEDIR, f'SlavicNER_synth_injection/evt_sents.xlsx'), engine='openpyxl')[lang].values.astype(str)
    evt_entities = pd.read_excel(os.path.join(BASEDIR, f'SlavicNER_synth_injection/evt_names.xlsx'), engine='openpyxl')[lang].values.astype(str)

    generated_pro_sentences, generated_pro_tags = create_synthetic_dataset_s2(
        pro_sentences, pro_entities, '<PRO>', 'PRO', num_samples_per_sentence=22
    )

    generated_evt_sentences, generated_evt_tags = create_synthetic_dataset_s2(
        evt_sentences, evt_entities, '<EVT>', 'EVT', num_samples_per_sentence=22
    )

    synth_sentences = generated_pro_sentences + generated_evt_sentences
    synth_tags = generated_pro_tags + generated_evt_tags
    return synth_sentences, synth_tags


def get_wikiner_dataset():
    """
    read WikiNER gold dataset from text files for all languages
    :return: Dict[str] - WikiNER dataset for specific language. Fields - texts: List[List[str]], tags: List[List[str]]
    """
    wikiner_langs = ['ru', 'be', 'bg', 'sl', 'uk', 'pl', 'cs', 'en']
    wikiner = {}

    for lang in wikiner_langs:
        wikiner_data = pd.read_excel(os.path.join(BASEDIR, f'./wikiner_gold/gold_{lang}.xlsx'), engine='openpyxl')
        sentences = []
        tags = []

        sentence_text = []
        sentence_tags = []
        for ind, row in wikiner_data.iterrows():

            if type(row['token']) != str and not math.isnan(row['token']):
                wikiner_data.loc[ind]['token'] = str(wikiner_data.loc[ind]['token'])
            if not row['label'] or (type(row['label']) is not str and math.isnan(row['label'])) or len(row['label']) > 6:
                wikiner_data.loc[ind]['label'] = 'O'

            if type(row['token']) != str and math.isnan(row['token']):
                sentences.append(sentence_text)
                tags.append(sentence_tags)
                sentence_text = []
                sentence_tags = []
            else:
                sentence_text.append(row['token'])
                sentence_tags.append(row['label'])

        tags = refactor_tags_to_bio_format(tags)
        wikiner[lang] = {
            'texts': sentences,
            'tags': tags
        }
    return wikiner


def get_custom_dataset(data_path: str):
    """
    Input format: csv with two columns. token<tab>tag, sentences separated with empty line. Tags are in BIO format.
    :param data_path: absolute path to dataset
    :return: conll_sentences: List[List[str]], conll_tags: List[List[str]]
    """
    conll_sentences, conll_tags = [], []
    sentence, entities = [], []
    with open(data_path, 'r') as custom_data:
        for line in custom_data:
            line = line.strip()
            if len(line) > 0:
                token, tag = line.split('\t')
                sentence.append(token)
                entities.append(tag)
            else:
                conll_sentences.append(sentence)
                conll_tags.append(entities)
                sentence = []
                entities = []
    return conll_sentences, conll_tags


def get_s1_data():
    """
    read S1 data from text files for all languages
    :return: Pandas Dataframes with locations, organisations, persons. Table format: columns - languages, rows - entities/contexts
    """
    locs_names = pd.read_csv(os.path.join(BASEDIR, './S1/LOC_names.csv'))
    locs_sentences = pd.read_csv(os.path.join(BASEDIR,'./S1/LOC_sents.csv'))

    orgs_names = pd.read_csv(os.path.join(BASEDIR,'./S1/ORG_239.csv'))
    orgs_sentences = pd.read_csv(os.path.join(BASEDIR,'./S1/ORG_sents.csv'))

    pers_names = pd.read_csv(os.path.join(BASEDIR,'./S1/PERS_7889_correct.csv'))
    pers_sentences = pd.read_csv(os.path.join(BASEDIR,'./S1/PER_sents.csv'))
    return locs_names, locs_sentences, orgs_names, orgs_sentences, pers_names, pers_sentences


def get_s2_data():
    """
    read S2 data from text files for all languages
    :return: Pandas Dataframes with locations, organisations, persons. Table format: columns - languages, rows - entities/contexts
    """
    locs_names = pd.read_csv(os.path.join(BASEDIR,'./S2/LOC_names.csv'))
    locs_sentences = pd.read_csv(os.path.join(BASEDIR,'./S2/LOC_sents.csv'))

    orgs_names = pd.read_csv(os.path.join(BASEDIR,'./S2/ORG_names.csv'))
    orgs_sentences = pd.read_csv(os.path.join(BASEDIR,'./S2/ORG_sents.csv'))

    pers_names = pd.read_csv(os.path.join(BASEDIR,'./S2/PER_names.csv'))
    pers_sentences = pd.read_csv(os.path.join(BASEDIR,'./S2/PER_sents.csv'))

    male_pers_names = pers_names[pers_names['gender']=='m']
    female_pers_names = pers_names[pers_names['gender']=='f']

    male_pers_names.index = range(len(male_pers_names))
    female_pers_names.index = range(len(female_pers_names))
    return locs_names, locs_sentences, orgs_names, orgs_sentences, male_pers_names, female_pers_names, pers_sentences


def get_s3_data():
    """
    read S3 data from text files for all languages
    :return: Pandas Dataframes with locations, organisations, persons. Table format: columns - languages, rows - entities/contexts
    orgs_names_dict: Dict[str]: organisations for every language with gender specification
    """
    locs_names = pd.read_csv(os.path.join(BASEDIR,'./S3/LOC/locs_v2.csv'))
    locs_sentences = pd.read_csv(os.path.join(BASEDIR,'./S2/LOC_sents.csv'))

    orgs_names = {
        'ru': pickle.load(open(os.path.join(BASEDIR,'./S3/ORG/orgs_v2/ru'), 'rb')),
        'uk': pickle.load(open(os.path.join(BASEDIR,'./S3/ORG/orgs_v2/uk'), 'rb')),
        'be': pickle.load(open(os.path.join(BASEDIR,'./S3/ORG/orgs_v2/be'), 'rb')),
        'sl': pickle.load(open(os.path.join(BASEDIR,'./S3/ORG/orgs_v2/sl'), 'rb')),
        'bg': pickle.load(open(os.path.join(BASEDIR,'./S3/ORG/orgs_v2/bg'), 'rb')),
        'cs': pickle.load(open(os.path.join(BASEDIR,'./S3/ORG/orgs_v2/cs'), 'rb')),
        'pl': pickle.load(open(os.path.join(BASEDIR,'./S3/ORG/ORG_pl_2000'), 'rb'))
    }
    orgs_sentences = pd.read_csv(open(os.path.join(BASEDIR,'./S3/ORG/ORG_sents.csv'), 'r'))

    pers_names = {
        'ru': pickle.load(open(os.path.join(BASEDIR,'./S3/PER/ru'), 'rb')),
        'uk': pickle.load(open(os.path.join(BASEDIR,'./S3/PER/uk'), 'rb')),
        'be': pickle.load(open(os.path.join(BASEDIR,'./S3/PER/be'), 'rb')),
        'sl': pickle.load(open(os.path.join(BASEDIR,'./S3/PER/sl'), 'rb')),
        'pl': pickle.load(open(os.path.join(BASEDIR,'./S3/PER/pl'), 'rb')),
        'cs': pickle.load(open(os.path.join(BASEDIR,'./S3/PER/cs'), 'rb')),
        'bg': pickle.load(open(os.path.join(BASEDIR,'./S3/PER/bg'), 'rb'))
    }
    pers_sentences = pd.read_csv(os.path.join(BASEDIR,'./S2/PER_sents.csv'))

    male_pers_names = {}
    female_pers_names = {}
    for lang in pers_names:
        male_pers_names[lang] = {'[male]': []}
        female_pers_names[lang] = {'[female]': []}

        for person in pers_names[lang]:
            if person[1] == 'm':
                male_pers_names[lang]['[male]'].append(person[0])
            elif person[1] == 'f':
                female_pers_names[lang]['[female]'].append(person[0])

    def preprocess_orgs_data(orgs_sentences, orgs_names):
        neutral_names, neutral_sentences = [], []
        plural_names, plural_sentences = [], []
        male_names, male_sentences = [], []
        female_names, female_sentences = [], []

        for name_entity in orgs_names:
            name, count, gender = name_entity

            if count == 'Plur':
                plural_names.append(name)
            elif gender == 'Masc':
                male_names.append(name)
            elif gender == 'Fem':
                female_names.append(name)
            else:
                neutral_names.append(name)

        return {
            '[n]': neutral_names,
            '[m]': male_names,
            '[pl]': plural_names,
            '[f]': female_names
        }

    orgs_names_dict = {}
    for lang in orgs_names:
        if lang == 'pl':
            orgs_names_dict[lang] = {
                '[n]': orgs_names['pl'],
                '[m]': orgs_names['pl'],
                '[pl]': orgs_names['pl'],
                '[f]': orgs_names['pl']
            }
            continue
        orgs_names_dict[lang] = preprocess_orgs_data(orgs_sentences[lang], orgs_names[lang])
    return locs_names, locs_sentences, orgs_names_dict, orgs_sentences, male_pers_names, female_pers_names, pers_sentences


def create_synthetic_dataset_s1(
        sentences: List[str],
        entities: List[str],
        search_tag: str,
        target_tag: str,
        num_samples_per_sentence: int = 10
):
    """
    generate NER dataset S1 from sentences and entities items
    :param sentences:
    :param entities:
    :param search_tag: special tag to search in context template and replace it with 'target_tag'
    :param target_tag: NER entity tag
    :param num_samples_per_sentence: number of sentences generated for every context example
    :return: dataset_sentences: List[List[str]], dataset_tags: List[List[str]]
    """
    dataset_sentences = []
    dataset_tags = []
    NUM_SAMPLES_PER_SENTENCE = num_samples_per_sentence

    for num, sentence in enumerate(sentences):
        if sentence.find(search_tag) != sentence.rfind(search_tag):
            continue

        for ind, word in enumerate(sentence.split()):
            if search_tag in word:
                entity_index = ind
                break
        else:
            continue

        for _ in range(NUM_SAMPLES_PER_SENTENCE):
            tags = ['O'] * len(sentence.split())

            random_entity = random.choice(entities).strip()
            random_entity_len = len(random_entity.split())
            if random_entity_len > 1:
                tags.extend(['O'] * (random_entity_len - 1))

                position_bias = 0
                if target_tag == 'LOC':
                    position_bias += 1

                for sub_ind in range(entity_index + position_bias, entity_index + random_entity_len):
                    tags[sub_ind] = target_tag
            else:
                tags[entity_index] = target_tag

            new_sentence = sentence.replace(search_tag, random_entity)
            dataset_sentences.append(new_sentence.split())
            dataset_tags.append(tags)

            if len(new_sentence.split()) != len(tags):
                print('error', num, random_entity, len(tags), len(new_sentence.split()))

    dataset_tags = refactor_tags_to_bio_format(dataset_tags)
    return dataset_sentences, dataset_tags


# TODO: combine S2 and S3 in single function
def create_synthetic_dataset_s2(
        sentences: List[str],
        entities: List[str],
        search_tag: str,
        target_tag: str,
        num_samples_per_sentence: int = 10
):
    """
    generate NER dataset S2 from sentences and entities items
    :param sentences:
    :param entities:
    :param search_tag: special tag to search in context template and replace it with 'target_tag'
    :param target_tag: NER entity tag
    :param num_samples_per_sentence: number of sentences generated for every context example
    :return: dataset_sentences: List[List[str]], dataset_tags: List[List[str]]
    """
    dataset_sentences = []
    dataset_tags = []
    NUM_SAMPLES_PER_SENTENCE = num_samples_per_sentence

    for num, sentence in enumerate(sentences):
        entities_indecies = []
        for ind, word in enumerate(sentence.split()):
            if search_tag in word:
                entities_indecies.append(ind)

        if len(entities_indecies) == 0:
            continue

        for _ in range(NUM_SAMPLES_PER_SENTENCE):
            tags = ['O'] * len(sentence.split())
            position_bias = 0
            new_sentence = sentence.split()

            for entity_index in entities_indecies:
                random_entity = random.choice(entities)
                if type(random_entity) != str and len(random_entity) > 1:
                    random_entity = random_entity[0]
                random_entity.strip()
                random_entity_len = len(random_entity.split())
                if random_entity_len > 1:
                    tags.extend(['O'] * (random_entity_len - 1))

                    for sub_ind in range(entity_index + position_bias,
                                         entity_index + position_bias + random_entity_len):
                        tags[sub_ind] = target_tag

                else:
                    tags[entity_index + position_bias] = target_tag

                new_sentence[entity_index + position_bias:entity_index + position_bias + 1] = random_entity.split()
                position_bias += random_entity_len - 1
            dataset_sentences.append(new_sentence)
            dataset_tags.append(tags)

    dataset_tags = refactor_tags_to_bio_format(dataset_tags)
    return dataset_sentences, dataset_tags


def create_synthetic_dataset_s3(
        sentences: List[str],
        entities: Dict[str, List[str]],
        search_tag: List[str],
        target_tag: str,
        num_samples_per_sentence: int = 10
):
    """
    generate NER dataset S3 from sentences and entities items
    :param sentences:
    :param entities: dict of possible entities. How to use, see src.train_xlm_model.train_wechsel_on_s3.
    :param search_tag: list of special tags to search in context template and replace it with 'target_tag'
    :param target_tag: NER entity tag
    :param num_samples_per_sentence: number of sentences generated for every context example
    :return: dataset_sentences: List[List[str]], dataset_tags: List[List[str]]
    """
    dataset_sentences = []
    dataset_tags = []
    NUM_SAMPLES_PER_SENTENCE = num_samples_per_sentence

    for num, sentence in enumerate(sentences):
        entities_indecies = []
        entities_values = []
        for ind, word in enumerate(sentence.split()):
            if word in search_tag:
                entities_indecies.append(ind)
                entities_values.append(word)
        if len(entities_indecies) == 0:
            continue

        for _ in range(NUM_SAMPLES_PER_SENTENCE):
            tags = ['O'] * len(sentence.split())
            position_bias = 0
            new_sentence = sentence.split()

            for ind, entity_index in enumerate(entities_indecies):
                entity_value = entities_values[ind]
                random_entity = random.choice(entities[entity_value])
                if type(random_entity) != str and len(random_entity) > 1:
                    random_entity = random_entity[0]
                random_entity.strip()
                random_entity_len = len(random_entity.split())
                if random_entity_len > 1:
                    tags.extend(['O'] * (random_entity_len - 1))

                    for sub_ind in range(entity_index + position_bias,
                                         entity_index + position_bias + random_entity_len):
                        tags[sub_ind] = target_tag

                else:
                    tags[entity_index + position_bias] = target_tag

                new_sentence[entity_index + position_bias:entity_index + position_bias + 1] = random_entity.split()
                position_bias += random_entity_len - 1
            dataset_sentences.append(new_sentence)
            dataset_tags.append(tags)

    dataset_tags = refactor_tags_to_bio_format(dataset_tags)
    return dataset_sentences, dataset_tags
