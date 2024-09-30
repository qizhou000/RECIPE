#%%
import threading, time
import numpy as np
import torch, os, json, re
from typing import Dict, List, Tuple, Union
from torch.nn.utils.rnn import pad_sequence
from copy import deepcopy
from utils.utils import set_tokenizer_pad_id
from transformers import  AutoTokenizer 
from datasets import load_dataset
from queue import Queue 
from collections import defaultdict
 

################################################################################
# A Parallel Dataset class: Preprocessing and generating data batches through  #
# sub processes.                                                               #
################################################################################
class ParallelDataset():
    def __init__(self, sample_count:int, get_data_by_ids_func,
        batch_size:Union[int, List[int]] = 256, shuffle = True, 
        buffer_size = 64, drop_last = False, random_seed = None) -> None:
        self.sample_count = sample_count
        self.set_batch_size(batch_size)
        # batch_size = [batch_size] if type(batch_size) == int else batch_size
        # self.batch_size = np.array([min(bs, sample_count) for bs in batch_size])
        self.shuffle = shuffle
        self.rng = np.random.default_rng(random_seed)
        self.select_ids = np.array(range(sample_count))
        if shuffle: 
            self.rng.shuffle(self.select_ids)
        self.drop_last = drop_last
        self.now_buffer_i = 0 # the idex of data has added into buffer
        self.now_yield_i = 0 # the idex of data has yielded
        self.buffer_size = buffer_size
        self.buffer = Queue()
        self.__get_data_by_ids__ = get_data_by_ids_func
        self.__fill_buffer__()

    def set_batch_size(self, batch_size:Union[int, List[int]]):
        if type(batch_size) != list and batch_size == 0:
            raise
        batch_size = [batch_size] if type(batch_size) != list else batch_size
        self.batch_size = np.array([min(bs, self.sample_count) for bs in batch_size])

    def __get_data_by_ids__(self, ids):
        raise

    def __fill_buffer__(self):
        def fill_buffer(): 
            while self.buffer.qsize() < self.buffer_size:
                bs = self.rng.choice(self.batch_size)
                tail_i = self.now_buffer_i + bs
                ids = self.select_ids[self.now_buffer_i:tail_i]
                if tail_i >= self.sample_count:
                    self.select_ids = np.array(range(self.sample_count))
                    if self.shuffle:
                        self.rng.shuffle(self.select_ids)
                    if tail_i > self.sample_count and self.drop_last:
                        self.now_buffer_i = 0
                        continue
                    self.now_buffer_i = tail_i - self.sample_count
                    extra_ids = self.select_ids[:self.now_buffer_i]
                    ids = np.concatenate([ids, extra_ids], 0)
                else:
                    self.now_buffer_i = tail_i
                d = self.__get_data_by_ids__(ids)
                self.buffer.put((d, len(ids)))
        for thread in threading.enumerate():
            if thread.name == 'data_filling':
                return
        threading.Thread(target = fill_buffer, name = 'data_filling').start() 
    
    def __len__(self): 
        if len(self.batch_size) > 1:
            print('The number of data batches is not accurate as `batch_size` is a list')
        bs = self.batch_size.mean()
        if self.drop_last:
            return int(np.floor(self.sample_count/bs))
        return int(np.ceil(self.sample_count/bs))

    def __iter__(self): 
        self.now_yield_i = 0
        return self

    def __next__(self):
        if self.now_yield_i >= self.sample_count:
            raise StopIteration
        if self.buffer.qsize() <= self.buffer_size/2:
            self.__fill_buffer__() 
        t = 0  
        while self.buffer.qsize() == 0:  
            print('\r', "Waiting data: %d s"%t, end='')
            time.sleep(1) 
            t += 1  
        d, data_n = self.buffer.get()
        self.now_yield_i += data_n
        return d


################################################################################
#    prompts & targets transform to input&output&mask token ids                # 
################################################################################
def prompts_target_to_x_y_mask(tokenizer, prompts:List[str], targets:List[str], device='cuda'):
    targets = deepcopy(targets)
    for i, t in enumerate(targets):
        targets[i] = t if t[0] == ' ' else ' ' + t
    input_ids, label_ids, masks = [], [], []
    for p, t in zip(prompts, targets):
        prompt_tok = tokenizer(p)['input_ids']
        input_tok = tokenizer(p + t, return_tensors="pt")['input_ids'][0]
        label_tok = input_tok.clone()[1:] 
        input_tok = input_tok[:-1] 
        mask = torch.ones_like(label_tok)
        mask[:len(prompt_tok)-1] *= 0
        input_ids.append(input_tok)
        label_ids.append(label_tok)
        masks.append(mask)
    input_ids = pad_sequence(input_ids, True, tokenizer.pad_token_id).to(device)
    label_ids = pad_sequence(label_ids, True, tokenizer.pad_token_id).to(device)
    masks = pad_sequence(masks, True, 0).to(device)
    return input_ids, label_ids, masks


################################################################################
#    prompts & predict length to get input&output&mask token ids               #  
################################################################################
def prompts_last_len_to_x_y_mask(tokenizer, prompts:List[str], pre_len:Union[int, float], 
        truncation = 1024, device='cuda'):
    input_ids, label_ids, masks = [], [], []
    for p in prompts:
        input_tok = tokenizer(p, return_tensors="pt")['input_ids'][0][:truncation]
        label_tok = input_tok.clone()[1:] 
        input_tok = input_tok[:-1] 
        mask = torch.zeros_like(label_tok)
        if type(pre_len) == int:
            mask[-pre_len:] += 1
        elif type(pre_len) == float and pre_len <= 1.:
            pl = int(len(mask) * pre_len)
            mask[-pl:] += 1
        else:
            raise
        input_ids.append(input_tok)
        label_ids.append(label_tok)
        masks.append(mask)
    input_ids = pad_sequence(input_ids, True, tokenizer.pad_token_id).to(device)
    label_ids = pad_sequence(label_ids, True, tokenizer.pad_token_id).to(device)
    masks = pad_sequence(masks, True, 0).to(device)
    return input_ids, label_ids, masks



################################################################################
#                     get structured test datasets                             #
################################################################################
class TestSampleList:
    '''
    Functions used to read and preprocess various datasets for evaluation,
    which return list with structure like [
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
    '''
    def load_and_select_data(path:str, test_i:Union[List, int], shuffle:bool, seed:int):
        with open(path, 'r') as f:
            data = json.load(f)
        idx = list(range(len(data)))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)
        if test_i == None:
            test_i = idx
        elif type(test_i) == int:
            test_i = idx[:test_i]
        elif type(test_i) == list:
            test_i = [idx[i] for i in test_i]
        else:
            raise
        return [data[i] for i in test_i]
 
    def zsre(path = 'data/evaluation/zsre/zsre_mend_eval.json', 
            test_i:Union[List, int] = None, shuffle = True, seed = 0):
        data = TestSampleList.load_and_select_data(path, test_i, shuffle, seed)
        test_sample_list = []
        for s in data:
            ns = {}
            ns['request'] = {
                'prompt': s['src'], 
                'target_new': s['alt'], 
                'subject': s['subject'],
                'ground_truth': s['answers'][0],
            }
            ns['generality'] = {
                'rephrase': [
                    {'prompt': s['rephrase'], 'target': s['alt']},
                ]
            }
            ns['locality'] = {
                'loc1': [
                    {'prompt': s['loc'], 'target': s['loc_ans']},
                ]
            }
            test_sample_list.append(ns)
        return test_sample_list

    def counterfact(path = 'data/evaluation/cf/counterfact-edit.json', 
            test_i:Union[List, int] = None, shuffle = True, seed = 0):
        # counterfact dataset path
        data = TestSampleList.load_and_select_data(path, test_i, shuffle, seed)
        test_sample_list = []
        for s in data:
            ns = {}
            ns['request'] = {
                'prompt': s['prompt'], 
                'target_new': s['target_new'], 
                'subject': s['subject'],
                'ground_truth': s['ground_truth'],
            }
            ns['generality'] = {
                'rephrase': [
                    {'prompt': s['rephrase_prompt'], 'target': s['target_new']},
                ]
            }
            ns['locality'] = {
                'loc1': [
                    {'prompt': s['locality_prompt'], 'target': s['locality_ground_truth']},
                ]
            }
            test_sample_list.append(ns)
        return test_sample_list

    def ripple_effect(path = 'data/evaluation/ripple_effect/ripe_test.json', 
                      test_i:Union[List, int] = None, shuffle = True, seed = 0):
        # ripple effect dataset path
        data = TestSampleList.load_and_select_data(path, test_i, shuffle, seed)
        test_sample_list = []
        for s in data:
            ns = {}
            ns['example_type'] = s['example_type']
            ns['request'] = {
                'prompt': s['prompt'], 
                'target_new': s['target_new'], 
                'subject': s['subject'],
            }
            if s['example_type'] == 'recent':
                ns['request']['ground_truth'] = s['target_new']
            else:
                ns['request']['ground_truth'] = s['ground_truth']
            gen_types = ['Logical_Generalization', 'Compositionality_I', 
                            'Compositionality_II', 'Subject_Aliasing']
            ns['generality'] = {}
            for gen_type in gen_types:
                ns['generality'][gen_type] = []
                for i in s[gen_type]:
                    for t in i['targets']:
                        if t != "":
                            ns['generality'][gen_type].append({'prompt': i['prompt'], 'target':t})
                            break
            loc_types = ['Relation_Specificity', 'Forgetfulness']
            ns['locality'] = {}
            for loc_type in loc_types:
                ns['locality'][loc_type] = []
                for i in s[loc_type]:
                    for t in i['targets']:
                        if t != "":
                            ns['locality'][loc_type].append({'prompt': i['prompt'], 'target':t})
                            break
            test_sample_list.append(ns)
        return test_sample_list


    def test_data(path, test_n = None):
        with open(path, 'r') as f:
            data = json.load(f)
        return data










 