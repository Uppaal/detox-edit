import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import json
import torch
import logging
import numpy as np
from copy import deepcopy
from utils.model_utils import (project_into_vocabluary, is_key, is_value,
                               get_lm_head, get_last_transformer_layer,
                               get_num_transformer_layers, get_hidden_dim, get_model_category)


class DeToxEdit():
    def __init__(self, model, tokenizer, pref_data_dps, centering=True, top_k_ranks=2, edit_layer_range=None, random_dps=True):

        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer

        self.model_category = get_model_category(model)  # 'gpt2' or 'llama' like architectures
        self.D = get_hidden_dim(model)  # Hidden dimension of the model
        self.num_layers = get_num_transformer_layers(model)
        self.E = get_lm_head(self.model)  # (V, D) for GPT-2

        self.pref_data_dps = pref_data_dps
        self.random_dps = random_dps
        self.centering = centering
        self.top_k_ranks = top_k_ranks
        if edit_layer_range is None:
            self.edit_layer_range = np.arange(self.num_layers)
        else:
            self.edit_layer_range = edit_layer_range


    def _load_preference_data(self):
        num_dps = self.pref_data_dps
        filedir = os.path.join(os.environ["DATASET_DIR"], 'toxicity_pairwise')
        filepath = os.path.join(filedir, 'split_0.jsonl')

        if not os.path.exists(filepath):
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

        preferred_data, non_preferred_data = [], []
        with open(filepath, 'r') as f:
            for line in f:
                preferred_data.append(json.loads(line)['unpert_gen_text'])
                non_preferred_data.append(json.loads(line)['pert_gen_text'])

        if num_dps != -1:  # 4096 points
            if not self.random_dps:
                preferred_data = preferred_data[:num_dps]
                non_preferred_data = non_preferred_data[:num_dps]
            else:
                indices = np.random.choice(len(preferred_data), num_dps, replace=False)
                preferred_data = [preferred_data[i] for i in indices]
                non_preferred_data = [non_preferred_data[i] for i in indices]
        logging.info(f'Loaded {len(preferred_data)} preferred and {len(non_preferred_data)} non-preferred samples.')

        preferred_inputs = self.tokenizer(preferred_data, return_tensors="pt", padding=True, truncation=True, max_length=128)  # 128, 37
        non_preferred_inputs = self.tokenizer(non_preferred_data, return_tensors="pt", padding=True, truncation=True, max_length=128)  # 128, 37

        return preferred_inputs, non_preferred_inputs


    def _get_hidden_sentence_embeddings(self, inputs):
        inputs = inputs.to(self.model.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        batch_size = min(50, input_ids.size(0))
        num_batches = inputs.input_ids.size(0) // batch_size
        sent_embs = []

        for i in range(num_batches):
            batch_input_ids = input_ids[i * batch_size: (i + 1) * batch_size]
            batch_attention_mask = attention_mask[i * batch_size: (i + 1) * batch_size]
            logging.info(f'Batch {i + 1}/{num_batches} of size {batch_input_ids.size(0)}')

            with torch.no_grad():
                outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # Tuple of len L tensors: (N, seq_len, D)
            del outputs
            hidden_states = hidden_states[1:]  # Remove the input layer embeddings
            hidden_states = torch.stack(hidden_states)  # (L, N, seq_len, D)

            last_layer = get_last_transformer_layer(self.model)
            penultimate_layer_embedding = hidden_states[-2]  # (N, seq_len, D)

            if self.model_category in ['gpt2', 'mistral', 'opt']:
                last_layer_emb = last_layer(penultimate_layer_embedding)[0]  # (N, seq_len, D)
            elif self.model_category == 'llama':
                inputs_embeds = self.model.model.embed_tokens(batch_input_ids)
                past_seen_tokens = 0
                cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
                causal_mask = self.model.model._update_causal_mask(batch_attention_mask, inputs_embeds, cache_position, past_seen_tokens)
                position_ids = cache_position.unsqueeze(0)
                last_layer_emb = last_layer(
                    penultimate_layer_embedding,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                )[0]
            elif self.model_category == 'gptj':
                last_layer_emb = hidden_states[-1]
            else:
                raise NotImplementedError(f'Model category not recognized: {self.model_category}')
            hidden_states[-1] = last_layer_emb

            hidden_sent_embs = torch.mean(hidden_states, dim=2)  # (L, N, D)
            sent_embs.append(hidden_sent_embs.detach().to('cpu'))
            del hidden_sent_embs, hidden_states
            torch.cuda.empty_cache()

        # sent_embs is a list of tensors of shape (L, N, D). Concatenate them along the batch dimension
        hidden_sent_embs = torch.cat(sent_embs, dim=1)  # (L, N, D)
        del sent_embs
        logging.info(f'Hidden sent: {hidden_sent_embs.shape}')
        torch.cuda.empty_cache()
        return hidden_sent_embs


    def _get_preference_matrix(self):
        preferred_inputs, non_preferred_inputs = self._load_preference_data()

        preferred_sent_embs = self._get_hidden_sentence_embeddings(preferred_inputs)  # (L, N, D)
        non_preferred_sent_embs = self._get_hidden_sentence_embeddings(non_preferred_inputs)  # (L, N, D)

        preference_matrix = (preferred_sent_embs - non_preferred_sent_embs) / 2  # (L, N, D)
        logging.info('Preference matrix calculated.')
        del non_preferred_sent_embs

        if self.centering:
            logging.info('Centering: Removing first singular vector from preference matrix.')

            for layer_num in range(preference_matrix.shape[0]):
                d = preference_matrix[layer_num].to(torch.float32)
                pref = deepcopy(preferred_sent_embs[layer_num].to(torch.float32))

                u, s, vt = torch.linalg.svd(pref, full_matrices=False)  # (N, D) -> (N, N), (N,), (N, D)
                projection_vector = vt[0].unsqueeze(dim=-1)  # (D, 1)
                P = projection_vector @ projection_vector.T  # (D, D)
                I = torch.eye(projection_vector.shape[0]).to(pref.device)  # (D, D)
                d = d @ (I - P)  # (N, D) @ (D, D) -> (N, D)
                preference_matrix[layer_num] = d.to(preference_matrix[layer_num].dtype) # d

        return preference_matrix


    def get_ats(self):

        preference_matrix = self._get_preference_matrix()  # (L, N, D)
        ats = {}

        for key in self.model.state_dict():
            if 'weight' in key and 'mlp' in key:
                layer_num = int(key.split('.')[2])  # Format: transformer.h.19.mlp.c_fc.weight
                ats[key] = preference_matrix[layer_num]
        return ats


    def svd_on_ats(self, ats):
        '''
        Key(D, 4D) -> U(D, D) S(D) V^T(D, 4D)
        Value(4D, D) -> U(4D, D) S(4D) V^T(D, D)
        x_l (N, D) -> U(N, N); S(N,); V^T(N, D)

        Note: v @ v.T is not numerically I, but plotting it as a heatmap shows that it is close to I.
        '''
        svd = {}
        for key in ats:
            logging.debug(f'Calculating SVD for: {key}')
            M = ats[key].to(torch.float32)  # SVD function only works with float32

            u, s, vt = torch.linalg.svd(M.cuda(), full_matrices=False)  # Skinny SVD, vt is V^T
            svd[key] = {'u': u.cpu(), 's': s.cpu(), 'v': vt.T.cpu()}
        logging.info('SVD of ATS calculated.')
        return svd


    def find_p_toxic(self, svd, rank_range=20):
        toxic_subspace = {}

        for key in svd.keys():
            layer_num = int(key.split('.')[2])  # Format: transformer.h.19.mlp.c_fc.weight
            if layer_num not in self.edit_layer_range:
                logging.info(f'Skipping layer {layer_num}')
                continue
            logging.info(f'Calculating toxic subspace for: {key}')

            singular_vectors = svd[key]['v']  # (D, N): N cols of (D,) vectors
            toxic_rank_list = np.arange(self.top_k_ranks)  # [0, 1] by default

            # Sum outer products of shortlisted ranks
            p_toxic = torch.zeros(self.D, self.D)
            for r in toxic_rank_list:
                singular_vector = singular_vectors[:, r].unsqueeze(dim=1)  # (D, 1)
                p_toxic += singular_vector @ singular_vector.T  # (D, 1) @ (1, D) -> (D, D)

                sorted_tokens = project_into_vocabluary(singular_vector.squeeze(), self.E.cpu(), self.tokenizer, top_k=10)
                logging.debug(f'Layer {layer_num} - Rank {r}: {" | ".join([x for x in sorted_tokens])}')

            toxic_subspace[key] = p_toxic
        logging.info('Toxic subspace calculated.')
        return toxic_subspace


    def edit_model(self, toxic_subspace, edit_keys=True, edit_values=True, layer_range=None):
        assert edit_keys or edit_values, 'At least one of edit_keys or edit_values should be True'
        logging.info(f'Editing keys: {edit_keys}, Editing values: {edit_values}.')

        if layer_range is None:
            layer_range = np.arange(get_num_transformer_layers(self.model))
        logging.info(f'Editing layers: {layer_range}')

        edited_state_dict = self.model.state_dict()
        for key in edited_state_dict:
            if key in toxic_subspace:

                layer_num = int(key.split('.')[2])
                if layer_num in layer_range:
                    logging.debug(f'Editing: {key}')
                    logging.debug(f'Module {key}: P_toxic mean: {toxic_subspace[key].mean()}.')

                    P_filter = torch.eye(self.D) - toxic_subspace[key]
                    P_filter = P_filter.to(edited_state_dict[key].device).to(self.model.dtype)

                    weight = edited_state_dict[key]
                    if self.model_category in ['llama', 'mistral', 'opt', 'gptj']:
                        weight = weight.T

                    if edit_keys and is_key(key, self.model_category):
                        modified_weight = P_filter @ weight  # (D, D) @ (D, 4D) -> (D, 4D)
                    elif edit_values and is_value(key, self.model_category):
                        modified_weight = weight @ P_filter  # (4D, D) @ (D, D) -> (4D, D)
                    else:
                        continue
                    if torch.allclose(weight, modified_weight) and ('gate_proj' not in key):
                        logging.warning(f'Module {key} not edited after projection.')

                    if self.model_category in ['llama', 'mistral', 'opt', 'gptj']:
                        modified_weight = modified_weight.T
                    edited_state_dict[key] = modified_weight.to('cuda').contiguous()  # contiguous for saving to disk

        self.model.load_state_dict(edited_state_dict, assign=True)
        logging.info('Edited model created.')
        return self.model


    def setup_for_edits(self):
        ats = self.get_ats()
        svd = self.svd_on_ats(ats)
        del ats
        self.toxic_subspace = self.find_p_toxic(svd)
        del svd
        torch.cuda.empty_cache()


    def apply_edit_end_to_end(self, edit_keys=True, edit_values=True, layer_range=None):
        # Measure speed and memory use
        import time
        import psutil
        import pynvml
        start_time = time.time()
        before_memory = psutil.virtual_memory().used
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        before_gpu_memory_used = info.used

        # Find P_toxic
        self.setup_for_edits()

        # Apply edit
        edited_model = self.edit_model(self.toxic_subspace, edit_keys, edit_values, layer_range)
        torch.cuda.empty_cache()

        end_time = time.time()
        time.sleep(1)
        after_memory = psutil.virtual_memory().used
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        after_gpu_memory_used = info.used
        print(f"Elapsed time: {end_time - start_time} seconds")
        print(f"System Memory Used: {(after_memory - before_memory) / (1024 * 1024)} MB")
        print(f"GPU Memory Used: {(after_gpu_memory_used - before_gpu_memory_used) / (1024 ** 2)} MB")

        return edited_model
