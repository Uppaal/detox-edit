import os
import json
import random
import logging
import numpy as np
from datasets import load_dataset as load_dataset_hf


def download_toxicity_preference(filedir):
    filepath = os.path.join(filedir, 'split_0.jsonl')
    logging.info('Preference data not found. Downloading...')
    os.makedirs(filedir, exist_ok=True)
    url = 'https://drive.google.com/file/d/1BmBkhNS4R2z5UwqVu5GhaTvFFeWOJfPC/view?usp=drive_link'
    os.system(f'gdown --id {url.split("/")[-2]} -O {filedir}/toxicity_pairwise.zip')  # Download file from google drive
    os.system(f'unzip {filedir}/toxicity_pairwise.zip -d {filedir}')  # Unzip the file
    os.system(f'rm {filedir}/toxicity_pairwise.zip')  # Delete the zip file
    os.system(f'mv {filedir}/toxicity_pairwise/* {filedir}')  # Move the files in the subdirectory to the parent directory
    os.system(f'rm -r {filedir}/toxicity_pairwise')  # Delete the subdirectory
    assert os.path.exists(filepath), 'Preference data download failed.'
    logging.info('Done.')


def reduce_num_dps(data, num_dps, random_dps):
    if num_dps != -1:  # 4096 points
        if not random_dps:
            data = data[:num_dps]
        else:
            logging.info('Shuffling data...')
            indices = np.random.choice(len(data), num_dps, replace=False)
            data = [data[i] for i in indices]
    return data


def load_toxicity_preference(filedir, num_dps=-1, random_dps=True):
    filepath = os.path.join(filedir, 'split_0.jsonl')
    if not os.path.exists(filepath):
        download_toxicity_preference(filedir)

    preferred_data, non_preferred_data = [], []
    with open(filepath, 'r') as f:
        for line in f:
            preferred_data.append(json.loads(line)['unpert_gen_text'])
            non_preferred_data.append(json.loads(line)['pert_gen_text'])

    # 4096 dps by default. Load fewer if needed.
    preferred_data = reduce_num_dps(preferred_data, num_dps, random_dps)
    non_preferred_data = reduce_num_dps(non_preferred_data, num_dps, random_dps)
    logging.info(f'Loaded toxicity preference dataset: {len(preferred_data)} preferred and {len(non_preferred_data)} non-preferred samples.')

    return preferred_data, non_preferred_data


def load_hh_golden_preference(num_dps=-1, random_dps=True):
    from detoxify import Detoxify
    toxicity_scorer = Detoxify('original', device='cuda')
    raw_data = load_dataset_hf('nz/anthropic-hh-golden-rlhf')['train']
    num_shortlisted_dps = 0
    i = 0
    preferred_data, non_preferred_data = [], []
    while num_shortlisted_dps <= num_dps and i < len(raw_data['prompt']):
        if toxicity_scorer.predict(raw_data['prompt'][i])['toxicity'] < 0.8:
            num_shortlisted_dps += 1
            preferred_data.append(f"{raw_data['prompt'][i]} {raw_data['chosen'][i]}")
            non_preferred_data.append(f"{raw_data['prompt'][i]} {raw_data['rejected'][i]}")
        i += 1

    preferred_data = reduce_num_dps(preferred_data, num_dps, random_dps)
    non_preferred_data = reduce_num_dps(non_preferred_data, num_dps, random_dps)
    logging.info(f'Loaded HH-Golden dataset: {len(preferred_data)} preferred and {len(non_preferred_data)} non-preferred samples.')
    return preferred_data, non_preferred_data


def load_hh_golden_challenge(num_dps=-1):
    raw_data = load_dataset_hf('nz/anthropic-hh-golden-rlhf')['train']
    num_dps = len(raw_data) if num_dps == -1 else num_dps
    challenge_prompts = [raw_data['prompt'][i] for i in range(num_dps)]
    return challenge_prompts


def load_safe_rlhf_preference(harm_category, num_dps=-1, shuffle=True, get_from_end=False):
    raw_data = load_dataset_hf('PKU-Alignment/PKU-SafeRLHF')
    harm_categories = list(raw_data['train'][0]['response_1_harm_category'].keys())
    assert harm_category in harm_categories, f'Harm category {harm_category} is not in the dataset. Must be one of:\n{harm_categories}'
    pref_data_full = []
    for dp in raw_data['train']:
        if dp['is_response_0_safe'] != dp['is_response_1_safe']:
            new_dp = {'prompt': dp['prompt']}
            if dp['is_response_0_safe']:
                new_dp['good_response'] = dp['response_0']
                new_dp['bad_response'] = dp['response_1']
                new_dp['category'] = [x for x in harm_categories if dp['response_1_harm_category'][x]]
            else:
                new_dp['good_response'] = dp['response_1']
                new_dp['bad_response'] = dp['response_0']
                new_dp['category'] = [x for x in harm_categories if dp['response_0_harm_category'][x]]
            pref_data_full.append(new_dp)

    logging.info(f'Dataset loaded. :{len(pref_data_full)} dps')
    logging.info(f'Loading {num_dps} dps for category: {harm_category}')

    if shuffle:
        random.shuffle(pref_data_full)
    dps = []
    if get_from_end:
        pref_data_full.reverse()
    for dp in pref_data_full:
        if harm_category in dp['category']:
            dps.append(dp)
        if len(dps) == num_dps and num_dps != -1:
            break

    preferred_data, non_preferred_data = [], []
    for i in range(len(dps)):
        preferred_data.append(f'HUMAN: {dps[i]["prompt"]}\nASSISTANT:{dps[i]["good_response"]}')
        non_preferred_data.append(f'HUMAN: {dps[i]["prompt"]}\nASSISTANT:{dps[i]["bad_response"]}')

    logging.info(f'Loaded Safe-RLHF ({harm_category}) dataset: {len(preferred_data)} preferred and {len(non_preferred_data)} non-preferred samples.')
    return preferred_data, non_preferred_data


def load_safe_rlhf_challenge(num_dps, harm_category):
    raw_data = load_dataset_hf('PKU-Alignment/PKU-SafeRLHF')
    harm_categories = list(raw_data['train'][0]['response_1_harm_category'].keys())
    assert harm_category in harm_categories, f'Harm category {harm_category} is not in the dataset. Must be one of:\n{harm_categories}'

    dps = []
    for dp in raw_data['test']:
        if harm_category in dp['response_0_harm_category'].keys() or harm_category in dp['response_1_harm_category'].keys():
            dps.append(dp)
        if len(dps) == num_dps and num_dps != -1:
            break

    challenge_prompts = [x['prompt'] for x in dps]
    return challenge_prompts