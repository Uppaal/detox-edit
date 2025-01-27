# Weight Editing for Toxicity Reduction

<!-- This repository contains the code for the paper "[DeTox: Toxic Subspace Projection for Model Editing](https://arxiv.org/abs/2405.13967)" (2024). -->

### Setup

- Create a virtual environment with python 3.9, using `requirements.txt`
  ````
  conda create -n <env_name> python=3.9
  conda activate <env_name>
  pip install -r requirements.txt
  ````

- Depnding on which model you want to run, pick the corresponding config file in `configs/`. Then change the filepaths in your config file to match your local setup.


### Editing the Model
- Edit the config file to set the parameters for the projection edit method.
   - `cuda_visible_devices` is a comma-separated list of GPU ids to use.
   - Model configurations:
     - `model_name`: Currently supports `gpt2`, `mistral`, `zephyr-sft`, `opt`, `gptj`
     - `save_edited_model`: If True, saves the edited model. 
     - `save_model_name`: Str
   - Dataset configurations:
     - `toxicity_task`: If False, aligns to a harmlesness dataset with multiple preferences, instead of the toxicity preference data.
     - `harmful_dataset`: Either [`Safe-RLHF`](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF)  or [`HH-Golden`](https://huggingface.co/datasets/nz/anthropic-hh-golden-rlhf)
     - `harm_category`: For `Safe-RLHF` dataset only. Specifies which kind of harm preference to edit for.
   - Configurations to find P_toxic:
     - `pref_data_dps`: How many datapoints to use for calculating the preference matrices
     - `centering`: If True, the preference matrix is projected away from the first singular vector of the preferred embeddings
  - Edit configurations:
    - `edit_keys`: If True, edits the keys of the MLP layer (not recommended, does not reduce toxicity)
    - `edit_values`: If True, edits the values of the MLP layer
    - `lowest_layer_to_edit`: The lowest layer to edit (zero indexed). If -1, all layers are edited
    - `highest_layer_to_edit`: The highest layer to edit. If -1, all layers are edited
  - Evaluation configurations:
    - `return_perplexity`: If True, returns the perplexity of the edited model on the data
    - `return_toxicity`: If True, returns the toxicity of the edited model on the data
    - `return_sample_generations`: If True, returns the generations of the edited model on 3 samples
  - Keys:
    - `hf_token`: Your token for the HuggingFace model hub. Required to access Mistral models.
    - `azure_openai_endpoint` and `azure_openai_api_key`: Required to calculate win-rate using GPT-4 Azure services.

- The file `detox.py` contains the edit method. To apply this and evaluate, run the following command:
````
python baselines/detox_edit.py -- config_file <name_of_config_file>
````
For example, if you want to edit the GPT-2 model, run:
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


### Citation

If you find our work useful, please cite our paper:
````
@inproceedings{uppaal2024profs,
  title={Model editing as a robust and denoised variant of DPO: A case study on toxicity},
  author={Uppaal, Rheeya and Dey, Apratim and He, Yiting and Zhong, Yiqiao and Hu, Junjie},
  booktitle={The Thirteenth International Conference on Learning Representations 2025},
  year={2024}
}
````

We use the preference and evaluation data from:
````
@article{lee2024mechanistic,
  title={A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity},
  author={Lee, Andrew and Bai, Xiaoyan and Pres, Itamar and Wattenberg, Martin and Kummerfeld, Jonathan K and Mihalcea, Rada},
  journal={arXiv preprint arXiv:2401.01967},
  year={2024}
}
````
