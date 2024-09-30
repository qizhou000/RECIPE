#%%
from editors.recipe.data import RECIPETrainData
from utils.utils import get_model_editor_config_path, model_path_map
from transformers import  AutoTokenizer, AutoModelForCausalLM
from editors.recipe.recipe import RECIPE, RECIPEConfig
from utils.global_attrs import ROOT_PATH
import os

def train_recipe(model_name:str, data_name, device, load_ckpt_path):
    model_path, config_path = get_model_editor_config_path(model_name, 'recipe')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map = device) 
    config = RECIPEConfig.from_yaml(config_path)
    recipe = RECIPE(model, tokenizer, config, device, model_path_map['roberta-base'])
    rtd = RECIPETrainData(tokenizer, None, data_name, None, False, device)  
    recipe.train_init(rtd.sample_count, rtd.get_data_by_ids, 
        batch_size = 8, 
        records_dir = os.path.join(ROOT_PATH, 'train_records'), 
        train_name_prefix = None, 
        train_name = None, 
        load_ckpt_path = load_ckpt_path, 
        save_ckpt_per_i = 3000, 
        log_per_i = 10, 
        random_seed = 1)
    recipe.train(1000) 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-mn', '--model_name', type=str)
    parser.add_argument('-dn', '--data_name', type=str)
    parser.add_argument('-ckpt', '--checkpoint', type=str, default = None)
    args = parser.parse_args()
    print(args)
    train_recipe(args.model_name, args.data_name, 'cuda:0', args.checkpoint)
