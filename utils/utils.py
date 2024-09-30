from transformers import  AutoTokenizer
from typing import List
import os
from .global_attrs import MODEL_PATH

model_path_map = {
    'gpt-j-6b': os.path.join(MODEL_PATH, 'gpt-j-6b'), 
    'gpt2-xl': os.path.join(MODEL_PATH, 'gpt2-xl'), 
    'llama-2-7b': os.path.join(MODEL_PATH, 'llama-2-7b-hf'),
    'llama-2-7b-chat': os.path.join(MODEL_PATH, 'Llama-2-7b-chat-hf'),
    'roberta-base': os.path.join(MODEL_PATH, 'roberta-base'), 
    'multi-qa-mpnet-base-dot-v1': os.path.join(MODEL_PATH, 'multi-qa-mpnet-base-dot-v1')
}

def set_tokenizer_pad_id(tokenizer:AutoTokenizer):
    if tokenizer.pad_token_id == None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print('Set [pad_token] as [eos_token].')

def get_model_editor_config_path(model_name:str, editor_name:str):
    model_name = model_name.lower()
    if 'gptj' in model_name or 'gpt-j' in model_name:
        config_name = 'gpt-j-6b.yaml'
        model_path = model_path_map['gpt-j-6b']
    elif 'llama' in model_name:
        config_name = 'llama-7b.yaml'
        model_path = model_path_map['llama-2-7b']
    elif 'llama' in model_name and 'chat' in model_name:
        config_name = 'llama-7b-chat.yaml'
        model_path = model_path_map['llama-2-7b-chat']
    elif 'gpt2' in model_name:
        config_name = 'gpt2-xl.yaml'
        model_path = model_path_map['gpt2-xl']
    else:
        raise
    config_path = os.path.join('configs', editor_name, config_name)
    return model_path, config_path

def get_editor(editor_name:str, edit_model_name:str, device:int, editor_ckpt_path = None):
    from transformers import  AutoTokenizer, AutoModelForCausalLM
    editor_name = editor_name.lower() 
    model_path, config_path = get_model_editor_config_path(edit_model_name, editor_name)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map = device) 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    set_tokenizer_pad_id(tokenizer)
    # get editor
    if editor_name == 'recipe':
        from editors.recipe import RECIPE, RECIPEConfig
        config = RECIPEConfig.from_yaml(config_path)
        editor = RECIPE(model, tokenizer, config, device, model_path_map['roberta-base'])
        if editor_ckpt_path != None:
            editor.load_ckpt(editor_ckpt_path, True, False)
    else:
        raise
    if device == 'auto':
        print('Set auto device as `cuda:0`')
        editor.device = 'cuda:0'
    return editor
