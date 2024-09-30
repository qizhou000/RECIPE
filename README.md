# Before training or testing RECIPE
First, please set the root directory path and model directory path in `utils/global_attrs.py`.
Then, please place the Huggingface weight file of the language model to be edited into `models/` directory.

# Train RECIPE
Please run:
```
python train_recipe.py -mn 'gpt2-xl' -dn 'zsre'  
```
Checkpoints will be saved in `train_records/recipe/gpt2-xl/train_name/checkpoints/`.
You can view training information in `train_records/recipe/gpt2-xl/train_name/logs/` through Tensorboard.

# Evaluate RECIPE
Please run:
```
python test_recipe.py -en 'recipe' -mn 'gpt2-xl' -et 'sequential' -dvc 'cuda:0' -ckpt 'train_records/recipe/gpt2-xl/train_name/checkpoints/a_checkpoint' -dn 'zsre' -edn 1000 
```
You can check results in `eval_results/recipe`.
