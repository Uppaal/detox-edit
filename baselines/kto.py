import os
import sys
import inspect
import argparse
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils.startup import main
parser = argparse.ArgumentParser(description='DeTox')
parser.add_argument('--config_file', default='gpt2-medium.ini', type=str, help='Config Filename. E.g. gpt2-medium.ini')

args = parser.parse_args()
config_filename = args.config_file
config = main(config_filename=config_filename)

import json
import logging
from trl import KTOConfig, KTOTrainer
from datasets import Dataset, DatasetDict
from utils.model_utils import load_large_model
from evaluation.evaluate_model import evaluate_model
from utils.dataset_utils import load_toxicity_preference


def format_preference_to_kto_style(pref, non_pref):
    def split_common_prefix(str1, str2):
        common_prefix = ""
        for ch1, ch2 in zip(str1, str2):
            if ch1 == ch2:
                common_prefix += ch1
            else:
                break

        continuation1 = str1[len(common_prefix):]
        continuation2 = str2[len(common_prefix):]
        return common_prefix, continuation1, continuation2

    formatted_data = []
    for i in range(num_dps):
        prompt, cont_good, cont_bad = split_common_prefix(pref[i], non_pref[i])
        dp = {'prompt': [{"content": prompt, "role": "user"}]}
        if i % 2 == 0:
            dp['completion'] = [{"content": cont_good, "role": "assistant"}]
            dp['label'] = True
        else:
            dp['completion'] = [{"content": cont_bad, "role": "assistant"}]
            dp['label'] = False
        formatted_data.append(dp)

    dataset = Dataset.from_list(formatted_data)
    return dataset


num_dps = config.getint('P_toxic Hyperparameters', 'pref_data_dps')
filedir = os.path.join(os.environ["DATASET_DIR"], 'toxicity_pairwise')
preferred_data, non_preferred_data = load_toxicity_preference(filedir, num_dps=num_dps, random_dps=True)
train_dataset = format_preference_to_kto_style(preferred_data[:num_dps], non_preferred_data[:num_dps])
logging.info('Dataset created.')

# Retrieve chat template
_, tokenizer = load_large_model('zephyr') # This is Mistral's DPO version
chat_template = tokenizer.chat_template

# Load model
model_id = config.get('Model', 'model_name')  # 'gpt2', 'llama', 'mistral', 'zephyr-sft'
model, tokenizer = load_large_model(model_id, quantize=True, add_peft=True)
if tokenizer.chat_template is None:
    tokenizer.chat_template = chat_template
    logging.info('Chat template created.')
logging.info('Model loaded.')

# Train model with KTO
training_args = KTOConfig(output_dir="KTO", logging_steps=10,
                          per_device_train_batch_size=2, per_device_eval_batch_size=1)
trainer = KTOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()
logging.info('Model trained.')

# Evaluate model
ppl, tox =  evaluate_model(model, tokenizer,
                           return_toxicity=config.getboolean('Evaluation', 'return_toxicity'),
                           return_perplexity=config.getboolean('Evaluation', 'return_perplexity'),
                           display_gen=config.getboolean('Evaluation', 'return_sample_generations'),
                           prompts=None)
logging.info(f'{model_id} - Perplexity: {ppl}, Toxicity: {tox}')
logging.info('Done.')
