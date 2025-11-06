<h1 style="text-align:center;">Projection Filter on Subspaces: Editing for Toxicity and Safety</h1>

<style>
.button-group {margin-top: 2.5em; text-align: center;}
.button {
  display: inline-flex;            /* side-by-side layout */
  align-items: center;             /* vertically center icon + text */
  gap: 10px;                       /* space between icon and text */
  background-color: #333;
  color: #fff;
  border-radius: 40px;             /* smooth pill shape */
  padding: 10px 20px;
  font-size: 1em;
  text-decoration: none;
  transition: background 0.2s ease;
}
.button-arxiv:hover { background-color: #B31B1B; color: #FFFFFF}
.button-huggingface:hover { background-color: #f1c232; color: #FFFFFF}
.button-blog:hover { background-color: #1DA1F2; color: #FFFFFF}
</style>

<div class="button-group">
  <a class="button button-arxiv" href="https://arxiv.org/abs/2405.13967">
      <img src="assets/arxiv-logo.svg" width="30" height="30" loading="lazy" decoding="async">
      <span>arXiv</span>
  </a>
  <a class="button button-blog" href="https://uppaal.github.io/projects/profs/profs.html">
      <img src="assets/web-logo.svg" width="30" height="30" loading="lazy" decoding="async">
      <span>Project Webpage</span>
  </a>
  <a class="button button-huggingface" href="https://huggingface.co/collections/Uppaal/profs">
      <img src="assets/hf-logo.svg" width="30" height="30" loading="lazy" decoding="async">
      <span>Checkpoints</span>
  </a>
</div>


## Paper
This repository provides the implementation and checkpoints used in [Model Editing as a Robust and Denoised variant of DPO: A Case Study on Toxicity](https://arxiv.org/abs/2405.13967) (ICLR 2025). 
You may also find an earlier version of this paper titled [DeTox: Toxic Subspace Projection for Model Editing](https://arxiv.org/abs/2405.13967). This is the same paper — apologies for any confusion!



<p style="text-align:center;">
<img src="assets/ProFS%20Method.png" alt="drawing" width="900"><br>
<i><b>Figure.</b> 
Schematic of ProFS (previously called DeTox). Toxic directions (in red) are projected out of the model’s MLP-value matrices, leaving other representational directions intact. </i>
</p>


## Checkpoints

If you would like to use our edited models without having to run our code, you can directly download our checkpoints from our [HuggingFace collection](https://huggingface.co/collections/Uppaal/profs).

| Base Model                                                                 | Edited for | Checkpoint                                                       |
|----------------------------------------------------------------------------|------------|------------------------------------------------------------------|
| [GPT-2 Medium](https://huggingface.co/openai-community/gpt2-medium)        | Toxicity   | [Link](https://huggingface.co/Uppaal/gpt2-medium-ProFS-toxicity) |
| [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6b)                     | Toxicity   | [Link](https://huggingface.co/Uppaal/gpt-j-6B-ProFS-toxicity)    |
| [OPT 6.7B](https://huggingface.co/facebook/opt-6.7b)                                                               | Toxicity   | [Link](https://huggingface.co/Uppaal/opt-6.7b-ProFS-toxicity)    |

[//]: # (| [Mistal 7B]&#40;https://huggingface.co/mistralai/Mistral-7B-v0.1&#41;              | Toxicity   | |)

[//]: # (| [Mistral-SFT 7B]&#40;https://huggingface.co/HuggingFaceH4/mistral-7b-sft-beta&#41; | Toxicity   |)

[//]: # (|                                                                            | Safety     |)


## Using this Codebase

### Setup

- Create a virtual conda environment, using `requirements.txt`
  ````
  conda create -n <env_name> python=3.10
  conda activate <env_name>
  pip install -r requirements.txt
  ````

- Depending on which model you want to run, pick the corresponding config file in `configs/`. Then change the filepaths in your config file to match your local setup. 
Each config file requires the following filepaths:
  -  `project_root`: The path to this repository. `<path_to_repository>/detox-edit` 
  - `dataset_dir`: All datasets used by the code will be automatically downloaded and stored here.
  - `hf_home`: All HuggingFace artifacts used by the code will be downloaded here.
  
### Editing the Model
- Edit the config file to set the parameters for the projection edit method.
   - `cuda_visible_devices` is a comma-separated list of GPU ids to use. For example `cuda_visible_devices = 3`
   - Model configurations:
     - `model_name`: Currently supports `gpt2`, `mistral`, `zephyr-sft`, `opt`, `gptj`.
     - `save_edited_model`: If True, saves the edited model. 
     - `save_model_name`: Name to save model with. These are saved in the `<project root>/checkpoints` directory.
   - Dataset configurations:
     - `toxicity_task`:
       - If True, uses the toxicity preference data from [Lee et. al, 2024](https://github.com/ajyl/dpo_toxic).
       - If False, aligns to a safety/harmlessness dataset with multiple and more complex preferences.
     - `harmful_dataset`: Used if `toxicity_task=False`. Either [`Safe-RLHF`](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF)  or [`HH-Golden`](https://huggingface.co/datasets/nz/anthropic-hh-golden-rlhf).
     - `harm_category`: Used if `toxicity_task=False` and `harmful_dataset=Safe-RLHF`. For a more fine-grained edit, this specifies which kind of harm preference to edit for.
   - Configurations to find P_toxic:
     - `pref_data_dps`: How many datapoints to use for calculating the preference matrices.
     - `centering`: If True, the preference matrix is projected away from the first singular vector of the preferred embeddings.
  - Edit configurations:
    - `edit_keys`: If True, edits the keys of the MLP layer (not recommended, as this does not reduce unwanted behaviour).
    - `edit_values`: If True, edits the values of the MLP layer.
    - `lowest_layer_to_edit`: The lowest layer to edit (zero indexed). If -1, all layers are edited.
    - `highest_layer_to_edit`: The highest layer to edit. If -1, all layers are edited.
  - Evaluation configurations:
    - `return_perplexity`: If True, returns the perplexity of the edited model on held out data.
    - `return_toxicity`: If True, returns the toxicity of the edited model on held out data.
    - `return_sample_generations`: If True, returns the generations of the edited model on 3 samples.
  - Keys:
    - `hf_token`: Your token for the HuggingFace model hub. Required to access Mistral models.
    - `azure_openai_endpoint` and `azure_openai_api_key`: Required to calculate win-rate using GPT-4 Azure services. Only required if `toxicity_task=False`.

- The file `detox.py` contains the edit method. To apply this and evaluate, run the following command:
````
python baselines/detox_edit.py --config_file <name_of_config_file>
````
For example, if you want to edit the GPT-2 model fill out the above values in `configs/gpt2-medium.ini`, and run:
````
python baselines/detox_edit.py --config_file gpt2-medium.ini
````
The script will print the results to the console.


### Running Alternate Baselines
We compare our method against the following baselines:
- [DPO](https://arxiv.org/abs/2305.18290)
- [KTO](https://arxiv.org/abs/2402.01306)
- [DexPerts](https://aclanthology.org/2021.acl-long.522/)
- [ToxRev](https://arxiv.org/abs/2310.09573)

For each baseline, we either include our implementation in `baselines/` or use the implementation of the authors. To run a specific baseline, 
````
python baselines/<baseline_of_your_choice>.py --config_file <name_of_config_file>
````

### Additional Evaluations

In addition to Toxicity and Perplexity, our models are also evaluated for general capability - across 7 EleutherAI LM Harness tasks: BoolQ, RTE, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, and OpenBookQA. To run this evaluation, fill in the model checkpoint path in `evaluation\evaluate_capability_zero_shot.sh` and run
````
bash evaluation/evaluate_capability_zero_shot.sh 
````

If training/editing on safety tasks where the toxicity scoring API ([Detoxify](https://github.com/unitaryai/detoxify)) cannot be used, we instead measure win rate as assessed by GPT-4. To run this evaluation, 
```
python evaluation/win_rate.py --model_1_id mistral --model_2_id mistral-edited --dataset_name HH-Golden --num_eval_dps 500 --max_new_tokens 100
```
`model_1_id` and `model_2_id` can be either the name of a saved checkpoint in `<project root>/checkpoints`; or if you want to use an off-the-shelf model, use one of `gpt2`, `mistral`, `zephyr-sft`, `opt`, `gptj`.

The win rate (of the second model over the first) will be printed to the console.

### Citation

If you find our work useful, please cite our paper:
````
@inproceedings{uppaal2025profs,
  title={Model editing as a robust and denoised variant of DPO: A case study on toxicity},
  author={Uppaal, Rheeya and Dey, Apratim and He, Yiting and Zhong, Yiqiao and Hu, Junjie},
  booktitle={The Thirteenth International Conference on Learning Representations 2025},
  year={2025}
}
````

[//]: # (We use the preference and evaluation data from:)

[//]: # (````)

[//]: # (@article{lee2024mechanistic,)

[//]: # (  title={A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity},)

[//]: # (  author={Lee, Andrew and Bai, Xiaoyan and Pres, Itamar and Wattenberg, Martin and Kummerfeld, Jonathan K and Mihalcea, Rada},)

[//]: # (  journal={arXiv preprint arXiv:2401.01967},)

[//]: # (  year={2024})

[//]: # (})

[//]: # (````)
