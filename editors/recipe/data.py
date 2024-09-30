#%%
import numpy as np
import os, json, re, torch
from typing import Dict, List, Union
from utils.utils import set_tokenizer_pad_id
from utils.global_attrs import ROOT_PATH
from transformers import  AutoTokenizer 
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.nn.utils.rnn import pad_sequence
from utils.data import prompts_last_len_to_x_y_mask
from tqdm import tqdm

def get_sim_matrix(query_n, sim_per_query, not_sim_per_query):
    t1 = [torch.eye(query_n) for i in range(sim_per_query)]
    t2 = [torch.zeros(query_n, query_n) for i in range(not_sim_per_query)]
    t1.extend(t2)
    m = torch.stack(t1, 2).reshape(query_n, query_n * (sim_per_query + not_sim_per_query))
    return m

def pt2xym(tokenizer, prompt:str, target:str):
    target = target if target[0] == ' ' else ' ' + target
    prompt_ids = tokenizer(prompt, return_tensors = 'pt').input_ids[0]
    all_ids = tokenizer(prompt + target, return_tensors = 'pt').input_ids[0]
    label_ids = all_ids[1:]
    input_ids = all_ids[:-1]
    mask = torch.zeros(len(label_ids))
    mask[len(prompt_ids)-1:] += 1
    return input_ids, label_ids, mask

def stack_xym(tokenizer, x_list, y_list, m_list, device):
    x = pad_sequence(x_list, True, tokenizer.pad_token_id).to(device)
    y = pad_sequence(y_list, True, tokenizer.pad_token_id).to(device)
    m = pad_sequence(m_list, True, 0).to(device)
    return x, y, m

class RECIPETrainData():
    def __init__(self, tokenizer:AutoTokenizer, data_n = None, data_name = 'cf', 
            data_path = None, wiki_for_loc = False, device='cuda', seed = 0) -> None:
        set_tokenizer_pad_id(tokenizer)
        if data_name.lower() in ['cf', 'counterfact']:
            if data_path == None:
                data_path = os.path.join(ROOT_PATH, 'data/meta-train/cf/counterfact-train.json')
            self.sample_count, self.get_data_by_ids = self.__cf__(tokenizer, data_n, data_path)
        elif data_name.lower() in ['zsre']:
            if data_path == None:
                data_path = os.path.join(ROOT_PATH, 'data/meta-train/zsre/zsre_mend_train.json')
            self.sample_count, self.get_data_by_ids = self.__zsre__(tokenizer, data_n, data_path)
        elif data_name.lower() in ['ripe', 'ripple effect']:
            if data_path == None:
                data_path = os.path.join(ROOT_PATH, 'data/meta-train/ripple_effect/ripe_train.json')
            self.sample_count, self.get_data_by_ids = self.__ripe__(tokenizer, data_n, data_path)
        elif data_name.lower() in ['mix', '__mix__']:
            if data_path == None:
                data_path = {
                    'cf': os.path.join(ROOT_PATH, 'data/meta-train/cf/counterfact-train.json'),
                    'zsre': os.path.join(ROOT_PATH, 'data/meta-train/zsre/zsre_mend_train.json'),
                    'ripe': os.path.join(ROOT_PATH, 'data/meta-train/ripple_effect/ripe_train.json')}
            cf_sample_count, cf_get_data_by_ids = self.__cf__(tokenizer, data_n, data_path['cf'])
            zsre_sample_count, zsre_get_data_by_ids = self.__zsre__(tokenizer, data_n, data_path['zsre'])
            ripe_sample_count, ripe_get_data_by_ids = self.__ripe__(tokenizer, data_n, data_path['ripe'])
            def mix_get_data_by_ids(ids:List[int]):
                r = self.rng.integers(0, 3)
                sample_count = [cf_sample_count, zsre_sample_count, ripe_sample_count][r]
                ids = self.rng.integers(0, sample_count, len(ids)) 
                return [cf_get_data_by_ids, zsre_get_data_by_ids, ripe_get_data_by_ids][r](ids)
            self.sample_count, self.get_data_by_ids = cf_sample_count+zsre_sample_count+ripe_sample_count, mix_get_data_by_ids
        else:
            raise KeyError
        self.tokenizer = tokenizer
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.wiki_for_loc = wiki_for_loc
        if wiki_for_loc:
            data_dir = os.path.join(ROOT_PATH, 'data/meta-train/comprehensive/wikitext/splited_sentences')
            self.random_wiki_s, self.random_wiki_xym = self.__wiki_splited__(tokenizer, data_dir = data_dir, device = device)

    # Counterfact
    def __cf__(self, tokenizer:AutoTokenizer, data_n, data_path):
        with open(data_path, 'r') as f: 
            data = json.load(f)
        sample_count = min(len(data), data_n) if data_n != None else len(data)
        relia_pt, gen_pt, loc_pt = [], [], []
        relia_xym, gen_xym, loc_xym = [], [], []
        for d in tqdm(data[:sample_count], 'Counterfact data preparing...'):
            relia_pt.append((d['prompt'], d['target_new']))
            relia_xym.append(pt2xym(tokenizer, d['prompt'], d['target_new']))
            gen_pt.append({'rephrase': [(d['rephrase_prompt'], d['target_new'])]})
            gen_xym.append({'rephrase': [pt2xym(tokenizer, d['rephrase_prompt'], d['target_new'])]})
            loc_pt.append({'original': [(d['locality_prompt'], d['locality_ground_truth'])]})
            loc_xym.append({'original': [pt2xym(tokenizer, d['locality_prompt'], d['locality_ground_truth'])]})
        get_data_by_ids = self.__get_data_by_ids_wrap__(relia_pt, gen_pt, loc_pt, relia_xym, gen_xym, loc_xym)
        return sample_count, get_data_by_ids

    # zsre
    def __zsre__(self, tokenizer:AutoTokenizer, data_n, data_path):
        with open(data_path, 'r') as f: 
            data = json.load(f)
        sample_count = min(len(data), data_n) if data_n != None else len(data)
        relia_pt, gen_pt, loc_pt = [], [], []
        relia_xym, gen_xym, loc_xym = [], [], []
        for d in tqdm(data[:sample_count], 'ZSRE data preparing...'):
            relia_pt.append((d['src'], d['alt']))
            relia_xym.append(pt2xym(tokenizer, d['src'], d['alt']))
            gen_pt.append({'rephrase': [(d['rephrase'], d['alt'])]})
            gen_xym.append({'rephrase': [pt2xym(tokenizer, d['rephrase'], d['alt'])]})
            loc_pt.append({'original': [(d['loc'], d['loc_ans'])]})
            loc_xym.append({'original': [pt2xym(tokenizer, d['loc'], d['loc_ans'])]})
        get_data_by_ids = self.__get_data_by_ids_wrap__(relia_pt, gen_pt, loc_pt, relia_xym, gen_xym, loc_xym)
        return sample_count, get_data_by_ids

    # ripe
    def __ripe__(self, tokenizer:AutoTokenizer, data_n, data_path):
        def get_pt_xym_from_a_type(type_data_list:List[Dict[str, Union[str, List]]]):
            pts = []
            for pt in type_data_list:
                for t in pt['targets']:
                    if t != "":
                        pts.append((pt['prompt'], t))
                        break
            xym = [pt2xym(tokenizer, pt[0], pt[1]) for pt in pts]
            return pts, xym
        with open(data_path, 'r') as f: 
            data = json.load(f)
        gen_types = ['Logical_Generalization', 'Compositionality_I', 
                            'Compositionality_II', 'Subject_Aliasing']
        loc_types = ['Relation_Specificity', 'Forgetfulness']
        relia_pt, gen_pt, loc_pt = [], [], []
        relia_xym, gen_xym, loc_xym = [], [], []
        data_n = len(data) if data_n == None else data_n
        bar = tqdm(total = data_n, desc='Ripple Effect data preparing...')
        now_data_n = 0
        for d in data:
            new_gen_pt, new_gen_xym, new_loc_pt, new_loc_xym = {}, {}, {}, {}
            for gen_type in gen_types:
                pts, xym = get_pt_xym_from_a_type(d[gen_type])
                if pts != []:
                    new_gen_pt[gen_type] = pts
                    new_gen_xym[gen_type] = xym
            for loc_type in loc_types:
                pts, xym = get_pt_xym_from_a_type(d[loc_type])
                if pts != []:
                    new_loc_pt[loc_type] = pts
                    new_loc_xym[loc_type] = xym
            if len(new_gen_pt) != 0 and len(new_loc_pt) != 0:
                relia_pt.append((d['prompt'], d['target_new']))
                relia_xym.append(pt2xym(tokenizer, d['prompt'], d['target_new']))
                gen_pt.append(new_gen_pt), gen_xym.append(new_gen_xym)
                loc_pt.append(new_loc_pt), loc_xym.append(new_loc_xym)
                now_data_n += 1
                bar.update(1)
                if now_data_n >= data_n:
                    break 
        get_data_by_ids = self.__get_data_by_ids_wrap__(relia_pt, gen_pt, loc_pt, relia_xym, gen_xym, loc_xym)
        return len(relia_pt), get_data_by_ids

    def __wiki_splited__(self, tokenizer:AutoTokenizer, token_pre_len:Union[int, float] = 0.3, 
            token_truncation = 64, sentence_truncation = 128, device='cuda', 
            data_dir = 'data/meta-train/comprehensive/wikitext/splited_sentences'):
        import json
        assert os.path.isdir(data_dir) 
        set_tokenizer_pad_id(tokenizer)
        sentences = []
        for d_name in tqdm(os.listdir(data_dir), 'Preparing splited wiki...'):
            if '.json' not in d_name:
                continue
            data_path = os.path.join(data_dir, d_name)
            with open(data_path, 'r') as f:
                sentences.extend(json.load(f))
        sample_count = len(sentences)
        def random_xym(n):
            ids = self.rng.integers(0, sample_count, n)
            input_ids, label_ids, masks = prompts_last_len_to_x_y_mask(tokenizer, 
                [sentences[i]for i in ids], token_pre_len, token_truncation, device)
            return input_ids, label_ids, masks
        def random_sentences(n):
            res = [' '.join(sentences[i].split()[:sentence_truncation]) for i in 
                   self.rng.integers(0, sample_count, n)]
            return res
        return random_sentences, random_xym
            
    def __get_data_by_ids_wrap__(self, relia_pt, gen_pt, loc_pt, relia_xym, gen_xym, loc_xym):
        def get_data_by_ids(ids:List[int]):
            def random_select(d:Dict[str, List]):
                ks = list(d.keys())
                d = d[ks[self.rng.integers(0, len(ks))]]
                return d[self.rng.integers(0, len(d))]
            contra_knowl = [relia_pt[i][0] + ' ' + relia_pt[i][1] for i in ids]
            contra_q_rel = [relia_pt[i][0] for i in ids]
            contra_q_gen = [random_select(gen_pt[i])[0] for i in ids]
            contra_q_loc = {}
            contra_q_loc['original_loc'] = [random_select(loc_pt[i])[0] for i in ids]
            if self.wiki_for_loc:
                contra_q_loc['wiki_loc'] = self.random_wiki_s(len(ids))
            # reliability data
            x_list, y_list, m_list = [[relia_xym[i][j] for i in ids] for j in range(3)]
            batch_relia_xym = stack_xym(self.tokenizer, x_list, y_list, m_list, self.device)
            # generality data
            x_list, y_list, m_list = [], [], []
            for i in ids:
                xym = random_select(gen_xym[i])
                x_list.append(xym[0])
                y_list.append(xym[1])
                m_list.append(xym[2])
            batch_gen_xym = {'original': stack_xym(self.tokenizer, x_list, y_list, m_list, self.device)}
            # locality data
            x_list, y_list, m_list = [], [], []
            for i in ids:
                xym = random_select(loc_xym[i])
                x_list.append(xym[0])
                y_list.append(xym[1])
                m_list.append(xym[2])
            batch_loc_xym = {'original_loc': stack_xym(self.tokenizer, x_list, y_list, m_list, self.device)}
            if self.wiki_for_loc:
                batch_loc_xym['wiki_loc'] = self.random_wiki_xym(len(ids))
            contra_data = (contra_knowl, contra_q_rel, contra_q_gen, contra_q_loc)
            edit_data = (batch_relia_xym, batch_gen_xym, batch_loc_xym)
            return contra_data, edit_data
        return get_data_by_ids
