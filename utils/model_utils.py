import os
import torch
import logging
import numpy as np
from peft.peft_model import PeftModelForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.mistral.modeling_mistral import MistralForCausalLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
logging.getLogger().setLevel(logging.INFO)


MODEL_IDENITFIER = {
    'gpt2': 'gpt2-medium',
    'mistral': 'mistralai/Mistral-7B-v0.1',
    'zephyr-sft': 'HuggingFaceH4/mistral-7b-sft-beta',
    'gptj': 'EleutherAI/gpt-j-6B',
    'opt': 'facebook/opt-6.7b',
}


def load_large_model(model_id, quantize=False, add_peft=False, hf_token=None):
    """
    Load a language model from HuggingFace.
    :param model_id: Name of the model from the MODEL_IDENTIFIER dictionary E.g. 'gpt2', 'mistral', 'zephyr-sft', 'gptj', 'opt'
    :param quantize: If True, quantize the model to 4-bit
    :param add_peft: If True, add LoRA with rank 64 to the model
    :param hf_token: Token for HuggingFace model hub. Required to access Mistral models.
    :return:
    """
    if hf_token is None:
        hf_token = os.environ['HF_TOKEN']

    model_path = MODEL_IDENITFIER[model_id]
    dtype = torch.float32 if model_id == 'gpt2' else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512,
                                              cache_dir=os.path.join(os.environ['HF_HOME'], 'hub'), token=hf_token)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=dtype,  # Non quantized weights are torch.float16 by default
        cache_dir=os.path.join(os.environ['HF_HOME'], 'hub'),
        token=hf_token,
        quantization_config=quantization_config,
    )

    if add_peft:
        model = prepare_model_for_kbit_training(model)  # preprocess the quantized model for training
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    model.max_length = tokenizer.model_max_length
    model.eval()

    logging.info(f'Model {model_id} loaded.')
    return model, tokenizer


def get_model_category(model):
    """
    Returns the category of the model.
    :param model: A language model subclassed by HuggingFace PreTrainedModel
    :return: Str
    """
    if isinstance(model, LlamaForCausalLM):
        return 'llama'
    if isinstance(model, MistralForCausalLM):
        return 'mistral'
    if isinstance(model, OPTForCausalLM):
        return 'opt'
    if isinstance(model, GPTJForCausalLM):
        return 'gptj'
    if isinstance(model, PeftModelForCausalLM):
        return get_model_category(model.model)
    if isinstance(model, GPT2LMHeadModel):
        return 'gpt2'

    raise ValueError('Unsupported model. Only GPT2 or LLaMA like architectures currently supported.')


def get_num_transformer_layers(model):
    """
    Returns the number of transformer layers in the model.
    :param model: A language model subclassed by HuggingFace PreTrainedModel
    :return: Int
    """
    model_category = get_model_category(model)
    if model_category in ['llama', 'mistral']:
        return len(model.model.layers)          # 32
    elif model_category == 'opt':
        return len(model.model.decoder.layers)  # 32
    elif model_category in ['gpt2', 'gptj']:
        return len(model.transformer.h)         # 24 / 28


def get_hidden_dim(model):
    """
    Returns the hidden dimension of the model.
    :param model: A language model subclassed by HuggingFace PreTrainedModel
    :return: Int
    """
    model_category = get_model_category(model)
    if model_category in ['llama', 'mistral', 'opt']:
        return model.config.hidden_size  # 4096
    else:
        return model.config.n_embd       # 1024 / 4096 for GPT-2 / GPT-J


def get_model_max_len(model):
    """
    Returns the maximum sequence length of the model.
    :param model: A language model subclassed by HuggingFace PreTrainedModel
    :return: Int
    """
    model_category = get_model_category(model)
    if model_category in ['gpt2', 'gptj']:
        return model.config.n_positions
    elif model_category in ['llama', 'mistral', 'opt']:
        return model.config.max_position_embeddings


def get_embedding_layer(model, return_weight=True):
    """
    Returns the embedding layer of the model.
    :param model: A language model subclassed by HuggingFace PreTrainedModel
    :param return_weight: If True, returns the weight of the embedding layer. If False, returns the module.
    :return: Module or Weight tensor
    """
    model_category = get_model_category(model)
    if model_category in ['llama', 'mistral']:
        layer = model.model.embed_tokens
    elif model_category == 'opt':
        layer = model.model.decoder.embed_tokens
    elif model_category in ['gpt2', 'gptj']:
        layer = model.transformer.wte
    if return_weight:
        return layer.weight
    return layer


def get_lm_head(model, return_weight=True):
    """
    Returns the language model head of the model.
    :param model: Language model subclassed by HuggingFace PreTrainedModel
    :param return_weight: If True, returns the weight of the head. If False, returns the module.
    :return: Module or Weight tensor
    """
    if return_weight:
        return model.lm_head.weight
    else:
        return model.lm_head


def get_last_transformer_layer(model):
    """
    Returns the last transformer layer of the model.
    :param model: Language model subclassed by HuggingFace PreTrainedModel
    :return: Transformer layer
    """
    model_type = get_model_category(model)
    if model_type in ['gpt2', 'gptj']:
        last_layer = model.transformer.h[-1]
    elif model_type in ['llama', 'mistral']:
        last_layer = model.model.layers[-1]
    elif model_type == 'opt':
        last_layer = model.model.decoder.layers[-1]
    else:
        raise ValueError('Unsupported model type: ', model_type)
    return last_layer


def is_key(module_name, model_category):
    """
    Check if the module is a MLP-key module.
    :param module_name: Name of the layer module. E.g. transformer.h.19.mlp.c_fc.weight
    :param model_category: Type of model, str
    :return: Boolean
    """
    if model_category == 'gpt2':
        return 'c_fc' in module_name
    elif model_category == 'gptj':
        return 'fc_in' in module_name
    elif model_category in ['llama', 'mistral']:
        return 'up_proj' in module_name
    elif model_category == 'opt':
        return 'fc1' in module_name
    else:
        raise ValueError('Unsupported model category: ', model_category)


def is_value(module_name, model_category):
    """
    Check if the module is a MLP-value module.
    :param module_name: Name of the layer module. E.g. transformer.h.19.mlp.c_fc.weight
    :param model_category: Type of model, str
    :return: Boolean
    """
    if model_category == 'gpt2':
        return 'c_proj' in module_name
    elif model_category == 'gptj':
        return 'fc_out' in module_name
    elif model_category in ['llama', 'mistral']:
        return 'down_proj' in module_name
    elif model_category == 'opt':
        return 'fc2' in module_name
    else:
        raise ValueError('Unsupported model category: ', model_category)


def get_layer_num(module_name, model_category):
    """
    Get the layer number from the module name.
    :param module_name: Name of the layer module. E.g. transformer.h.19.mlp.c_fc.weight
    :param model_category: Type of model, str
    :return: Layer number, int
    """
    if model_category in ['gpt2', 'llama', 'mistral', 'gptj']:
        return int(module_name.split('.')[2])
    elif model_category == 'opt':
        return int(module_name.split('.')[-3])
    else:
        raise ValueError('Unsupported model category: ', model_category)


def project_into_vocabluary(vector, E, tokenizer, top_k=20, bottom_k=-1):
    """
    Project a vector into the vocabulary space and return the top_k tokens.
    :param vector: D dimensional vector
    :param E: Language model embedding matrix (V, D)
    :param tokenizer: Model tokenizer
    :param top_k: How many top tokens to return
    :param bottom_k: How many bottom tokens to return. If -1, return top_k tokens
    :return:
    """
    vector = vector.to(torch.float32).to('cuda')
    E = E.to(torch.float32).to('cuda')
    vocab_ranking = torch.matmul(E, vector)     # (V,)
    sorted_token_ids = np.argsort(vocab_ranking.detach().cpu().numpy())[::-1]  # Descending order
    if bottom_k == -1:
        sorted_tokens = [tokenizer.decode(x).strip() for x in sorted_token_ids[:top_k]]
        logging.debug([(sorted_token_ids[i], sorted_tokens[i], vocab_ranking[sorted_token_ids[i]].item()) for i in range(top_k)])
    else :
        sorted_tokens = [tokenizer.decode(x).strip() for x in sorted_token_ids[-bottom_k:][::-1]]  # Least score to most score
    return sorted_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )