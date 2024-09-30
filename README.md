# RECIPE
Source code for EMNLP 2024 (main) paper *Lifelong Knowledge Editing for LLMs with Retrieval-Augmented Continuous Prompt Learning*.

## Before training or testing RECIPE
Please set the root directory path and model directory path in `utils/global_attrs.py`.
Then, please place the Huggingface weight file of the language model to be edited into `models/` directory.

## Train RECIPE
Please run:
```
python train_recipe.py -mn 'gpt2-xl' -dn 'zsre'  
```
Checkpoints will be saved in `train_records/recipe/gpt2-xl/train_name/checkpoints/`.
You can view training information in `train_records/recipe/gpt2-xl/train_name/logs/` through Tensorboard.

## Evaluate RECIPE
Please run:
```
python test_recipe.py -en 'recipe' -mn 'gpt2-xl' -et 'sequential' -dvc 'cuda:0' -ckpt 'train_records/recipe/gpt2-xl/train_name/checkpoints/a_checkpoint' -dn 'zsre' -edn 1000 
```
You can check results in `eval_results/recipe`.

## Citation
Please cite our paper if you use RECIPE in your work.
```bibtex
@inproceedings{chen2024recipe,
    title = {Lifelong Knowledge Editing for LLMs with Retrieval-Augmented Continuous Prompt Learning},
    author={Qizhou Chen and Taolin Zhang and Xiaofeng He and Dongyang Li and Chengyu Wang and Longtao Huang and Hui Xue},
    year = 2024,
    booktitle = {EMNLP},
    url = {https://2024.emnlp.org/program/accepted_main_conference/}
}
```