import torch, os, json
from editors.editor import BaseEditor
from datetime import datetime
import numpy as np
from copy import deepcopy
from utils.data import prompts_target_to_x_y_mask
from typing import List, Dict, Union
from tqdm import tqdm
from time import time
from collections import defaultdict
from transformers import AutoTokenizer
from editors.utils.generate import generate_fast
from .llm_general_eval import BaseLLMGeneralEval
import nltk, scipy 

class EditorEvaluation():
    def __init__(self, editor:BaseEditor, test_sample_list:List[Dict], 
        evaluation_name = None, results_dir = 'eval_results', ) -> None:
        '''
        This class is used to evaluate overall performance of editor. 
        `test_sample_list`: The list of test samples. The data structure is 
            assumed to be: [
                { # test1
                    'request': {'prompt': str, 'target_new': str, ...},
                    'generality': {
                        'gen_1_name':[
                            {'prompt': str, 'target': str, ...},
                            {'prompt': str, 'target': str, ...}, ...
                        ],
                        'gen_2_name':[
                            {'prompt': str, 'target': str, ...},
                            {'prompt': str, 'target': str, ...}, ...
                        ], ...
                    },
                    'locality': {
                        'loc_1_name':[
                            {'prompt': str, 'target': str, ...},
                            {'prompt': str, 'target': str, ...}, ...
                        ],
                        'loc_2_name':[
                            {'prompt': str, 'target': str, ...},
                            {'prompt': str, 'target': str, ...}, ...
                        ], ...
                    }
                }, 
                { # test2
                    'request':{'prompt': str, 'target_new': str, ...},
                    'generality': ...
                }, ...
            ].
        `results_dir` & `evaluation_name`: Used to create result directory.
            `evaluation_name` can be set as dataset name.
        '''
        self.editor = editor
        self.test_sample_list = test_sample_list
        editor_name, model_name = editor.name_of_editor_and_model()
        t = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
        evaluation_name = evaluation_name if evaluation_name else t
        self.result_dir = os.path.join(results_dir, editor_name, model_name, evaluation_name)
        print('Evaluation results directory: ', self.result_dir)

 
    def evaluate_single_edit(self, test_fluency = False):
        sample_list = deepcopy(self.test_sample_list)
        tok = self.editor.tokenizer
        model = self.editor.model
        device = self.editor.device
        print('Evaluating reliability, generality and locality for %s with single editing.'%self.editor.name_of_editor_and_model()[0])
        self.editor.restore_to_original_model()
        results = []
        for s in tqdm(sample_list, dynamic_ncols = True):
            # prepare data
            xym, bts = self.__preprocess_test_samples_before_edit__([s], model, tok, device)
            # edit
            start_t = time()
            self.editor.edit_one_piece(s['request'])
            bts['edit_time'] = time() - start_t
            # compute scores 
            results.append(self.__get_results_after_edit__(xym, bts, model, tok))
            # Restore to original model
            self.editor.restore_to_original_model()
        mean_results = self.__get_mean_results__(results)
        save_dir = os.path.join(self.result_dir, 'single_edit')
        # save results
        self.save_results(os.path.join(save_dir, 'results.json'), results)
        self.save_results(os.path.join(save_dir, 'mean_results.json'), mean_results)
        return results, mean_results
 
    def evaluate_batch_edit(self, batch_size = 32, if_random = False, random_seed = None, 
                            discard_last = True, test_fluency = False):
        if not self.editor.if_can_batch_edit():
            raise
        tok = self.editor.tokenizer
        model = self.editor.model
        device = self.editor.device
        sample_list = deepcopy(self.test_sample_list)
        if if_random:
            if random_seed == None:
                random_seed = int(time()*1000000) % 1000000
            np.random.default_rng(random_seed).shuffle(sample_list)
        print('Evaluating reliability, generality and locality for %s with batch editing.'%self.editor.name_of_editor_and_model()[0])
        self.editor.restore_to_original_model()
        results = []
        if discard_last:
            end_idx = len(sample_list) + 1 
        else:
            end_idx = int(np.ceil(len(sample_list)/batch_size) * batch_size + 1)
        for i in tqdm(range(batch_size, end_idx, batch_size), dynamic_ncols = True):
            # prepare data
            s = sample_list[i - batch_size:i]
            xym, bts = self.__preprocess_test_samples_before_edit__(s, model, tok, device)
            # edit
            start_t = time()
            self.editor.edit_batch(bts['reliability']['request'])
            bts['edit_time'] = time() - start_t
            # compute scores 
            results.append(self.__get_results_after_edit__(xym, bts, model, tok))
            # Restore to original model
            self.editor.restore_to_original_model()
        mean_results = self.__get_mean_results__(results)
        mean_results['batch_size'] = batch_size
        # save results
        save_dir = os.path.join(self.result_dir, 'batch_edit_'+str(batch_size), 
                        'seed_%d'%random_seed if if_random else 'non_random')
        self.save_results(os.path.join(save_dir, 'results.json'), results)
        self.save_results(os.path.join(save_dir, 'mean_results.json'), mean_results)
        return results, mean_results


    def evaluate_sequential_edit(self, sequential_edit_n = 1000,  if_random = False, 
                    random_seed = None, discard_last = True, test_fluency = False): 
        ''' Sequentially edit `sequential_edit_n` times and then test 
            reliability, generality and locality on edited samples.
        '''
        tok = self.editor.tokenizer
        model = self.editor.model
        device = self.editor.device
        sample_list = deepcopy(self.test_sample_list)
        if if_random:
            if random_seed == None:
                random_seed = int(time()*1000000) % 1000000
            np.random.default_rng(random_seed).shuffle(sample_list)
        print('Evaluating reliability, generality and locality for %s with %d sequential editing.'
              %(self.editor.name_of_editor_and_model()[0], sequential_edit_n))
        self.editor.restore_to_original_model()
        results = []
        if discard_last:
            end_idx = len(sample_list) + 1 
        else:
            end_idx = int(np.ceil(len(sample_list)/sequential_edit_n) * sequential_edit_n + 1)
        for seq_i in tqdm(range(sequential_edit_n, end_idx, sequential_edit_n), dynamic_ncols = True):
            # prepare data
            sl = sample_list[seq_i - sequential_edit_n:seq_i]
            test_samples = []
            for i, s in enumerate(tqdm(sl, leave = False, desc = "Prepare data", dynamic_ncols = True)): 
                xym, bts = self.__preprocess_test_samples_before_edit__([s], model, tok, device)
                test_samples.append((xym, bts, s))
            # Sequential edit
            for i, (xym, bts, s) in enumerate(tqdm(test_samples, leave = False, desc = "Sequential editing", dynamic_ncols = True)): 
                start_t = time()
                self.editor.edit_one_piece(s['request'])
                bts['edit_time'] = time() - start_t
                bts['edit_order'] = i + 1
            # compute scores 
            now_seq_results = []
            for i, (xym, bts, _) in enumerate(tqdm(test_samples, leave = False, desc = "Testing", dynamic_ncols = True)): 
                now_seq_results.append(self.__get_results_after_edit__(xym, bts, model, tok))
            results.append(now_seq_results)
            # Restore to original model after one sequential editing
            self.editor.restore_to_original_model()
        # compute overall mean results
        overall_mean_results = []
        for r in results:
            overall_mean_results.extend(r)
        overall_mean_results = self.__get_mean_results__(overall_mean_results)
        # save results
        save_dir = os.path.join(self.result_dir, 'sequential_edit_'+str(sequential_edit_n), 
                            'seed_%d'%random_seed if if_random else 'non_random')
        self.save_results(os.path.join(save_dir, 'results.json'), results)
        self.save_results(os.path.join(save_dir, 'overall_mean_results.json'), overall_mean_results) 
        return results, overall_mean_results


    def llm_general_after_sequential_edit(self, general_datasets:List[BaseLLMGeneralEval], 
        sequential_edit_n = 1000, if_random = False, random_seed = None, discard_last = True): 
        ''' Sequentially edit `sequential_edit_n` times and then test llm's general performance
        ''' 
        model = self.editor.model
        sample_list = deepcopy(self.test_sample_list)
        if if_random:
            if random_seed == None:
                random_seed = int(time()*1000000) % 1000000
            np.random.default_rng(random_seed).shuffle(sample_list)
        editor_name, model_name = self.editor.name_of_editor_and_model()
        print('Evaluating %s general performance after %s sequential editing %d times.'
              %(model_name, editor_name, sequential_edit_n))
        if discard_last:
            end_idx = len(sample_list) + 1 
        else:
            end_idx = int(np.ceil(len(sample_list)/sequential_edit_n) * sequential_edit_n + 1)
        results = defaultdict(float)
        test_n = 0
        for seq_i in tqdm(range(sequential_edit_n, end_idx, sequential_edit_n), dynamic_ncols = True):
            self.editor.restore_to_original_model()
            # prepare data
            sl = sample_list[seq_i - sequential_edit_n:seq_i]
            # Sequential edit
            for sample in tqdm(sl, leave = False, desc = "Sequential editing", dynamic_ncols = True): 
                self.editor.edit_one_piece(sample['request'])
            # compute scores 
            for d in general_datasets:
                assert d.model is model 
                score = d.test_llm_general_perform()
                results[d.dataset_name()] += score
            test_n += 1
        # compute overall mean results
        for k, v in results.items():
            results[k] /= test_n
        # save results
        save_dir = os.path.join(self.result_dir, 'sequential_edit_'+str(sequential_edit_n), 
                            'seed_%d'%random_seed if if_random else 'non_random')
        self.save_results(os.path.join(save_dir, 'llm_general.json'), results, 6)
        return results


    def __preprocess_test_samples_before_edit__(self, test_samples_list:List[Dict],
                                                model, tok, device):
        '''
        Input selected test samples with structure like `self.test_sample_list`.
        Return:
        `xym`: {
            'reliability': {
                'request': (input_ids, label_ids, masks)
            },
            'generality': {
                'gen_1_name': (input_ids, label_ids, masks), 
                'gen_2_name': (input_ids, label_ids, masks), ...
            },
            'locality': {
                'loc_1_name': (input_ids, predict_ids_before_edit, masks),
                'loc_2_name': (input_ids, predict_ids_before_edit, masks), ...
            }
        }
        `batched_test_samples`: {
            'reliability': {
                'request': [
                    {'prompt': str, 'target_new': str, ...},
                    {'prompt': str, 'target_new': str, ...}, ...
                ]
            },
            'generality': {
                'gen_1_name': [
                    {'prompt': str, 'target': str, ...},
                    {'prompt': str, 'target': str, ...}, ...
                ],
                'gen_2_name': [
                    {'prompt': str, 'target': str, ...},
                    {'prompt': str, 'target': str, ...}, ...
                ], ...
            },
            'locality': {
                'loc_1_name': [
                    {'prompt': str, 'target': str, 'predict_before_edit': str, ...},
                    {'prompt': str, 'target': str, 'predict_before_edit': str, ...}, ...
                ],
                'loc_2_name': [
                    {'prompt': str, 'target': str, 'predict_before_edit': str, ...},
                    {'prompt': str, 'target': str, 'predict_before_edit': str, ...}, ...
                ], ...
            }
        }
        '''
        xym_pt = defaultdict(lambda: defaultdict(lambda: defaultdict(list))) 
        batched_test_samples = defaultdict(lambda: defaultdict(list))
        for s in test_samples_list:
            xym_pt['reliability']['request']['prompts'].append(s['request']['prompt'])
            xym_pt['reliability']['request']['targets'].append(s['request']['target_new'])
            batched_test_samples['reliability']['request'].append(s['request'])
            for gen_name in s['generality'].keys():
                for g in s['generality'][gen_name]:
                    xym_pt['generality'][gen_name]['prompts'].append(g['prompt'])
                    xym_pt['generality'][gen_name]['targets'].append(g['target'])
                    batched_test_samples['generality'][gen_name].append(g)
            for loc_name in s['locality'].keys():
                for l in s['locality'][loc_name]:
                    xym_pt['locality'][loc_name]['prompts'].append(l['prompt'])
                    xym_pt['locality'][loc_name]['targets'].append(l['target'])
                    batched_test_samples['locality'][loc_name].append(l)
        xym = defaultdict(dict)
        xym['reliability']['request'] = prompts_target_to_x_y_mask(tok, 
                                xym_pt['reliability']['request']['prompts'], 
                                xym_pt['reliability']['request']['targets'], device)
        for gen_name in xym_pt['generality'].keys():
            xym['generality'][gen_name] = prompts_target_to_x_y_mask(tok, 
                                xym_pt['generality'][gen_name]['prompts'], 
                                xym_pt['generality'][gen_name]['targets'], device)
        for loc_name in xym_pt['locality'].keys():
            x, _, m = prompts_target_to_x_y_mask(tok, 
                                xym_pt['locality'][loc_name]['prompts'], 
                                xym_pt['locality'][loc_name]['targets'], device)
            with torch.no_grad(): # y: [batch_size, max_prompts&targets_token_n]
                y = torch.softmax(model(input_ids = x).logits, 2).argmax(2) 
            xym['locality'][loc_name] = (x, y, m)
            for r, ps in zip(batched_test_samples['locality'][loc_name],
                        [tok.decode(yi[mi.to(bool)]) for yi, mi in zip(y, m)]):
                r['predict_before_edit'] = ps
        return xym, batched_test_samples


    def __get_results_after_edit__(self, xym, bts, model, tok):
        '''
        xym/bts: prepared by `self.__preprocess_test_samples_before_edit__` 
        return results `bts`: {
            'indicator_1': {
                'mean_acc': float,
                'indicator_1_1': [
                    {'acc': float, 'predict_after_edit': str, Others: OthersType, ...},
                    {'acc': float, 'predict_after_edit': str, Others: OthersType, ...}, ...
                ], 'indicator_1_1_mean_acc': float,
                'indicator_1_2': [
                    {'acc': float, 'predict_after_edit': str, Others: OthersType, ...},
                    {'acc': float, 'predict_after_edit': str, Others: OthersType, ...}, ...
                ], 'indicator_1_2_mean_acc': float, 
                ...
            },
            'indicator_2': {...}, ...
        }
        '''
        for k1 in xym.keys(): # [reliability, generality, locality]
            k1_n = 0
            mean_acc_k1 = 0
            for k2 in xym[k1].keys():
                acc, predict_strs = accuracy_and_prediction(model, *xym[k1][k2], tok)
                for s, a, p in zip(bts[k1][k2], acc, predict_strs):
                    s['acc'] = float(a)
                    s['predict_after_edit'] = p
                bts[k1][k2+'_mean_acc'] = float(torch.mean(acc, 0))
                k1_n += len(acc)
                mean_acc_k1 += float(torch.sum(acc, 0))
            bts[k1]['mean_acc'] = mean_acc_k1/k1_n
        return bts
    
    def __get_mean_results__(self, results:list, indicators = ['reliability', 'generality', 'locality']):
        '''
        Assume `results` structure: [
            { # Sample 1
                'edit_time': flaot,
                'indicator_1': {
                    'indicator_1_1': float,
                    'indicator_1_2': float, 
                    'indicator_1_3': float, ...
                    'others': OtherType, ...
                }, 
                'indicator_2': {...}, ...
            },
            { # Sample 2 
                'edit_time': flaot, ...
            }, ...
        ]
        return `mean_results`: {
            'edit_time': flaot,
            'count': int,
            'indicator_1': {
                'indicator_1_1': float,
                'indicator_1_2': float, 
                'indicator_1_3': float, ...
            }, 
            'indicator_2': {...}, ...
        }
        '''
        mean_results = {i:defaultdict(float) for i in indicators}
        results_count = {i:defaultdict(float) for i in indicators}
        mean_results['edit_time'] = 0
        results_count['edit_time'] = 0
        for rs in results:
            mean_results['edit_time'] += rs['edit_time']
            results_count['edit_time'] += 1
            for k1 in indicators:
                for k2, v2 in rs[k1].items():
                    if type(v2) == float:
                        mean_results[k1][k2] += v2
                        results_count[k1][k2] += 1
        mean_results['edit_time'] /= results_count['edit_time']
        for k1 in indicators:
            for k2 in mean_results[k1].keys():
                mean_results[k1][k2] /= results_count[k1][k2]
        mean_results['count'] = len(results)
        return mean_results


    def save_results(self, save_path:str, results:Dict, decimal = 4):
        def set_decimal(r):
            if isinstance(r, list):
                for i in range(len(r)):
                    r[i] = set_decimal(r[i])
            elif isinstance(r, dict) or isinstance(r, defaultdict):
                for k in r.keys():
                    r[k] = set_decimal(r[k])
            elif isinstance(r, float):
                r = round(r, decimal)
            return r
        res = deepcopy(results)
        res = set_decimal(res)
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_path), 'w') as f:
            json.dump(res, f, indent = 4)


def accuracy_and_prediction(model, x:torch.Tensor, y:torch.Tensor, m:torch.Tensor, 
                        tokenizer:AutoTokenizer):
    with torch.no_grad(): # pre_y: [batch_size, max_prompts&targets_token_n]
        pre_y = torch.softmax(model(input_ids = x).logits, 2).argmax(2) 
    acc = ((pre_y == y) * m).sum(1)/m.sum(1) # [batch_size]
    predict_strs = [tokenizer.decode(yi[mi.to(bool)]) for yi, mi in zip(pre_y, m)]
    return acc, predict_strs

