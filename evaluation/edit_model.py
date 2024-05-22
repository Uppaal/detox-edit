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

import numpy as np
from detox import DeToxEdit
from evaluate_model import evaluate_model
from utils.model_utils import load_large_model



# Read configs - finding the toxic subspace
PREF_DATA_DPS = config.getint('P_toxic Hyperparameters', 'pref_data_dps')
CENTERING = config.getboolean('P_toxic Hyperparameters', 'centering')

# Read configs - which parts of model to edit
edit_keys = config.getboolean('Edit Configuration', 'edit_keys')
edit_values = config.getboolean('Edit Configuration', 'edit_values')
lower_layer = config.getint('Edit Configuration', 'lowest_layer_to_edit')
upper_layer = config.getint('Edit Configuration', 'highest_layer_to_edit')
top_k_ranks = config.getint('Edit Configuration', 'top_k_ranks')
if lower_layer == -1 or upper_layer == -1:
    layer_range = None
else:
    layer_range = np.arange(lower_layer, upper_layer)

# Read configs - evaluation
return_perplexity = config.getboolean('Evaluation', 'return_perplexity')
return_toxicity = config.getboolean('Evaluation', 'return_toxicity')
return_sample_generations = config.getboolean('Evaluation', 'return_sample_generations')


# Load model
model_id = config.get('Model', 'model_name')
model, tokenizer = load_large_model(model_id)

# Apply edit
editor = DeToxEdit(model=model, tokenizer=tokenizer,
                        pref_data_dps=PREF_DATA_DPS, centering=CENTERING,
                        top_k_ranks=top_k_ranks, edit_layer_range=layer_range,
                        random_dps=True)
edited_model = editor.apply_edit_end_to_end(edit_keys=edit_keys, edit_values=edit_values, layer_range=layer_range)


# Save the edited model
save_edited_model = config.getboolean('Model', 'save_edited_model')
if save_edited_model:
    savename = config.get('Model', 'save_model_name')
    savedir = os.path.join(os.environ["PROJECT_ROOT"], 'checkpoints')
    os.makedirs(savedir, exist_ok=True)
    edited_model.save_pretrained(os.path.join(savedir, savename))
    tokenizer.save_pretrained(os.path.join(savedir, savename))
    print(f'Saved edited model to {os.path.join(savedir, savename)}')


# Evaluate the edited model
ppl, tox = evaluate_model(edited_model, tokenizer,
               return_perplexity=return_perplexity, return_toxicity=return_toxicity, display_gen=return_sample_generations)
print(f'{model_id} - Perplexity: {ppl}, Toxicity: {tox}')
