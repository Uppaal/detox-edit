import os
import sys
import inspect
import logging
from tqdm import tqdm
from copy import deepcopy
from openai import AzureOpenAI

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils.dataset_utils import load_safe_rlhf_challenge, load_hh_golden_challenge



def llm_judge_eval(response_set_1, response_set_2, challenge_prompts):
    with open('evaluation/gpt4-eval-prompt.txt', 'r') as f:
        system_prompt = f.read()

    # Load LLM (GPT-4) Judge
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01"
    )

    judge_inputs = []
    for i in range(len(challenge_prompts)):
        dp = deepcopy(system_prompt)
        dp = dp.replace('{prompt}', challenge_prompts[i])
        dp = dp.replace('{answer_a}', response_set_1[i])
        dp = dp.replace('{answer_b}', response_set_2[i])
        judge_inputs.append(dp)

    responses = []
    for dp in tqdm(judge_inputs):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": dp},
                {"role": "user", "content": ""}])

        responses.append(response.choices[0].message.content)

    asst_b_win = 0
    for r in responses:
        if r is not None and r[0].isdigit():
            r = r.split('\n')[0].split(' ')
            if int(r[0]) <= int(r[1]):
                asst_b_win += 1

    logging.info(f'Second Model Win Rate: {asst_b_win / len(responses) * 100:.2f}%')
    return asst_b_win / len(responses) * 100


def get_continuation(prompts, model, tokenizer, max_new_tokens=100):
    continuations = []
    for prompt in tqdm(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        response = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens,
                                  pad_token_id=tokenizer.pad_token_id)  # (1, response_len)

        # We only want to measure toxicity for the generation, excluding the prompt
        response = response[0, len(input_ids[0]):].tolist()  # (response_len, ) list
        response = tokenizer.decode(response).strip()  # str

        continuations.append(response)
    return continuations


def calculate_win_rate(model_1, model_2, tokenizer, dataset_name, num_eval_dps, max_new_tokens=100, harm_category=None):

    # Load challenge prompts
    if dataset_name == 'Safe-RLHF':
        challenge_prompts = load_safe_rlhf_challenge(num_dps=num_eval_dps, harm_category=harm_category)
    elif dataset_name == 'HH-Golden':
        challenge_prompts = load_hh_golden_challenge(num_dps=num_eval_dps)
    else:
        raise ValueError(f'Dataset {dataset_name} not supported. Must be one of Safe-RLHF, HH-Golden')

    continuations_1 = get_continuation(challenge_prompts, model_1.cpu().to("cuda:0"), tokenizer, max_new_tokens=max_new_tokens)
    logging.info(f'Model 1 continuations done.')

    continuations_2 = get_continuation(challenge_prompts, model_2.cpu().to("cuda:0"), tokenizer, max_new_tokens=max_new_tokens)
    logging.info(f'Model 2 continuations done.')

    win_rate = llm_judge_eval(continuations_1, continuations_2, challenge_prompts)
    return win_rate
