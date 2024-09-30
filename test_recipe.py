#%%
import os, argparse, sys
from utils.utils import get_editor
from evaluation.editor_eval import EditorEvaluation
from utils.data import TestSampleList


def eval_multi_edit(editor, eval_data_name = 'ZSRE', data_path = None, 
        edit_type:str = 'sequential', edit_n:int = 10, data_sample_n:int = None, 
        shuffle = True, seed = 0, extra_evaluation_name = None):
    data_map = {
        'zsre': ['data/evaluation/zsre/zsre_mend_eval.json', TestSampleList.zsre],
        'cf': ['data/evaluation/cf/counterfact-edit.json', TestSampleList.counterfact],
        'ripe': ['data/evaluation/ripple_effect/ripe_test.json', TestSampleList.ripple_effect]
    }
    eval_data_name = eval_data_name.lower()
    assert eval_data_name in data_map.keys()
    data_path = data_map[eval_data_name][0] if data_path == None else data_path
    test_sample_list = data_map[eval_data_name][1](data_path, data_sample_n, shuffle, seed)
    evaluation_name = eval_data_name.upper()
    if extra_evaluation_name != None:
        evaluation_name += '-' + extra_evaluation_name
    ev = EditorEvaluation(editor, test_sample_list, evaluation_name) 
    if edit_n == 1:
        ev.evaluate_single_edit()
    else:
        if edit_type == 'sequential':
            ev.evaluate_sequential_edit(edit_n, True, seed) 
        elif edit_type == 'batch':
            ev.evaluate_batch_edit(edit_n, True, seed) 
        else:
            raise

def has_evaluated(editor_name:str, model_name:str, data_name:str, edit_n:int):
    editor_name = editor_name.lower()
    model_name = model_name.lower()
    if 'llama' in model_name:
        model_name = 'llama-7b'
    elif 'gpt-j' in model_name or 'gptj' in model_name:
        model_name = 'gpt-j-6b'
    elif 'gpt2' in model_name:
        model_name = 'gpt2-xl'
    else:
        raise
    if edit_n == 1:
        dir_name = 'single_edit'
    else:
        dir_name = 'sequential_edit_' + str(edit_n)
    path = os.path.join('eval_results', editor_name, model_name, data_name, dir_name)
    if os.path.exists(path):
        return True, path
    if edit_n != 1:
        dir_name = 'batch_edit_' + str(edit_n)
    path = os.path.join('eval_results', editor_name, model_name, data_name, dir_name)
    if os.path.exists(path):
        return True, path
    return False, path

def get_attr():
    parser = argparse.ArgumentParser()
    parser.add_argument('-en', '--editor_name', type=str, help='Editor name: FT, KN, MEMIT...', required=True)
    parser.add_argument('-mn', '--edit_model_name', type=str, help='Editing model name: GPT-J, LLAMA...', required=True)
    parser.add_argument('-et', '--edit_type', type=str, required=True, help='Such as: sequential, batch')
    parser.add_argument('-dvc', '--device', type=str, help='CUDA device for editing.', required=True)
    parser.add_argument('-ckpt', '--editor_ckpt_path', type=str, default = None, help='For Editors that needs training.')
    parser.add_argument('-dn', '--data_name', type=str, default = 'ZSRE', help = 'Evaluating dataset.')
    parser.add_argument('-edn', '--edit_n', type=int, default = 10, help = 'Sequential editing number.')
    parser.add_argument('-dsn', '--data_sample_n', type=int, default = None, help = 'Sample number for evaluation.')
    parser.add_argument('-sd', '--seed', type=int, default = 0, help = 'Random seed.')
    args = parser.parse_args()
    return args
 
if __name__ == '__main__':
    cfg = get_attr()
    he, path = has_evaluated(cfg.editor_name, cfg.edit_model_name, cfg.data_name, cfg.edit_n)
    if he:
        print('Has evaluated: ', path)
        sys.exit()
    print(cfg)
    editor = get_editor(cfg.editor_name, cfg.edit_model_name, cfg.device, cfg.editor_ckpt_path)
    eval_multi_edit(editor, cfg.data_name, None, cfg.edit_type, cfg.edit_n, cfg.data_sample_n, True, cfg.seed, None) 

 