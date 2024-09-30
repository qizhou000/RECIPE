from datasets import load_from_disk
from abc import ABC, abstractmethod
from utils.data import prompts_target_to_x_y_mask
from typing import List
import numpy as np
import torch, os
from tqdm import tqdm

class BaseLLMGeneralEval(ABC):
    '''
    Basic dataset class Used for test LLM's general performance with probability.
    '''
    def __init__(self, model, tokenizer, test_n, device, seed):
        import datasets
        assert datasets.__version__ == '2.18.0'
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.test_n = test_n
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    def shuffle_and_select_data(self):
        rng = np.random.default_rng(self.seed)
        rng.shuffle(self.data)
        self.data = self.data if self.test_n == None else self.data[:self.test_n]
    @abstractmethod
    def test_llm_general_perform(self):
        raise
    @abstractmethod
    def dataset_name(self):
        raise

class StrAnsLLMEval(BaseLLMGeneralEval):
    '''
    Basic dataset class for datasets evaluated with PPL of answer string.
    '''
    def __init__(self, model, tokenizer, test_n, device, seed):
        super().__init__(model, tokenizer, test_n, device, seed)
    @abstractmethod
    def get_xym(self, idxs:List[int]):
        # return input_ids:torch.Tensor, output_ids:torch.Tensor, masks_for_eval:torch.Tensor
        raise
    def test_llm_general_perform(self, tempe = 1)->float:
        self.model = self.model.eval().requires_grad_(False)
        sum_res = 0
        sum_res_n = 0
        for i in tqdm(range(len(self)), 'Testing on '+self.dataset_name()):
            x, y, m = self.get_xym([i])
            logit = self.model(input_ids = x).logits[0]
            m_ids = torch.nonzero(m[0])[:, 0]
            x = torch.stack([logit[mi] for mi in m_ids])
            y = torch.stack([y[0, mi] for mi in m_ids]).unsqueeze(-1)
            p = torch.softmax(x * tempe, 1)
            p = torch.gather(p, 1, y)
            ppl = torch.exp(-torch.mean(torch.log(p)))
            sum_res += 1/ppl
            sum_res_n += 1
        return float(sum_res/sum_res_n)


class SingleSelectAnsLLMEval(BaseLLMGeneralEval):
    '''
    Basic dataset class for datasets evaluated with normalized probability of options.
    '''
    def __init__(self, model, tokenizer, test_n, device, seed):
        super().__init__(model, tokenizer, test_n, device, seed)
    @abstractmethod
    def get_input_option(self, idx:int):
        # Assume the final token of input_ids predict the answer.
        # return input_ids:torch.Tensor, option_ids:List[int], ans_id:int
        raise
    def input_ops_ans_str2ids(self, input_str:str, ops:List[str], ans:str):
        input_ids = self.tokenizer(input_str, return_tensors="pt")['input_ids'].to(self.device)
        option_ids = [self.tokenizer.encode(op, add_special_tokens=False)[0] for op in ops]
        ans_id = self.tokenizer.encode(ans, add_special_tokens=False)[0]
        return input_ids, option_ids, ans_id
    def test_llm_general_perform(self)->float:
        self.model = self.model.eval().requires_grad_(False)
        hit_n = 0
        for i in tqdm(range(len(self)), 'Testing on '+self.dataset_name()):
            input_ids, option_ids, ans_id = self.get_input_option(i)
            logit = self.model(input_ids=input_ids).logits[0, -1][option_ids]
            p = torch.softmax(logit, 0)
            max_p_i = torch.argmax(p)
            ans_i = option_ids.index(ans_id)
            hit_n += max_p_i == ans_i
        return float(hit_n/len(self))

class SQUAD_V2(StrAnsLLMEval):
    """
    SQUAD_V2 is a dataset class for the Stanford Question Answering Dataset (SQuAD) version 2, which involves question-answering tasks.
    SQUAD_V2 dataset is loaded from huggingface datasets.
    Reference: https://huggingface.co/datasets/squad_v2

    Example data format:
    [{'id': '56ddde6b9a695914005b9628', 'title': 'Normans', 'context': 'The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.', 'question': 'In what country is Normandy located?', 'answers': {'text': ['France', 'France', 'France', 'France'], 'answer_start': [159, 159, 159, 159]}}, ...]
    """
    def __init__(self, model, tokenizer, test_n = None, dataset_path='data/llm_general/squad_v2', device = 'cuda', seed = 0):  
        super().__init__(model, tokenizer, test_n, device, seed)   
        self.data = []
        # data = load_dataset("squad_v2")["validation"]
        data = load_from_disk(dataset_path)["validation"]
        for d in data:
            self.data.append(d)
        self.prompt = 'Please read the context and find the answer to the question. If there is no answer, reply N/A.'
        self.shuffle_and_select_data()

    def dataset_name(self):
        return 'SQUAD_V2'

    def get_xym(self, idxs:List[int]):
        labels = [self.data[i]['answers']['text'][0] if len(self.data[i]['answers']['text']) != 0 else 'N/A' for i in idxs]
        contents = []
        for i in idxs:
            content = self.prompt + '\nTitle: ' + self.data[i]['title'] + '\n'
            content += 'Context: ' + self.data[i]['context']+'\n'
            content += 'Question: ' + self.data[i]['question']+'\nAnswer:'
            contents.append(content)
        xym = prompts_target_to_x_y_mask(self.tokenizer, contents, labels, self.device)
        return xym


class MMLU(SingleSelectAnsLLMEval):
    """
    MMLU is a dataset class for the Multimodal Multi-Task Learning Understanding dataset, covering various educational and professional fields.
    MMLU dataset is loaded from huggingface datasets: lukaemon/mmlu (test set).
    
    Reference: https://huggingface.co/datasets/cais/mmlu/viewer/abstract_algebra/test

    Example data format:
    [{'input': "This question refers to the following information.\nRead the the following quotation to answer questions.\nThe various modes of worship which prevailed in the Roman world were all considered by the people as equally true; by the philosopher as equally false; and by the magistrate as equally useful.\nEdward Gibbon, The Decline and Fall of the Roman Empire, 1776–1788\nGibbon's interpretation of the state of religious worship in ancient Rome could be summarized as", 'A': "In ancient Rome, religious worship was decentralized and tended to vary with one's social position.", 'B': 'In ancient Rome, religious worship was the source of much social tension and turmoil.', 'C': 'In ancient Rome, religious worship was homogeneous and highly centralized.', 'D': 'In ancient Rome, religious worship was revolutionized by the introduction of Christianity.', 'target': 'A', 'task': 'high_school_european_history'}, ...]
    """
    def __init__(self, model, tokenizer, test_n = None, dataset_path='data/llm_general/mmlu', device = 'cuda', seed = 0):  
        super().__init__(model, tokenizer, test_n, device, seed)   
        self.data = []
        self.tasks = ['high_school_european_history', 'business_ethics', 'clinical_knowledge', 'medical_genetics',
                    'high_school_us_history', 'high_school_physics', 'high_school_world_history', 'virology',
                    'high_school_microeconomics', 'econometrics', 'college_computer_science', 'high_school_biology',
                    'abstract_algebra', 'professional_accounting', 'philosophy', 'professional_medicine', 'nutrition',
                    'global_facts', 'machine_learning', 'security_studies', 'public_relations', 'professional_psychology',
                    'prehistory', 'anatomy', 'human_sexuality', 'college_medicine', 'high_school_government_and_politics',
                    'college_chemistry', 'logical_fallacies', 'high_school_geography', 'elementary_mathematics', 'human_aging',
                    'college_mathematics', 'high_school_psychology', 'formal_logic', 'high_school_statistics', 'international_law',
                    'high_school_mathematics', 'high_school_computer_science', 'conceptual_physics', 'miscellaneous', 'high_school_chemistry',
                    'marketing', 'professional_law', 'management', 'college_physics', 'jurisprudence', 'world_religions', 'sociology',
                    'us_foreign_policy', 'high_school_macroeconomics', 'computer_security', 'moral_scenarios', 'moral_disputes',
                    'electrical_engineering', 'astronomy', 'college_biology']

        for task in self.tasks:
            # data = load_dataset("cais/mmlu", task)["test"]
            data = load_from_disk(os.path.join(dataset_path, task))["test"]
            for d in data:
                d["task"] = task
                self.data.append(d)
        self.prompt = 'Please select the correct answer for the following questions:\n'
        self.shuffle_and_select_data()

    def dataset_name(self):
        return 'MMLU'
    
    def get_input_option(self, idx:int):
        ops = ['A','B','C','D','E','F','G','H','I']
        d = self.data[idx]
        input_str = self.prompt + d['question'] + '\n'
        questions = []
        options = []
        for j, c in enumerate(d['choices']):
            questions.append('(' + ops[j] + ') ' + c)
            options.append(ops[j])
        input_str += '\n'.join(questions) + '\nAnswer: ('
        return self.input_ops_ans_str2ids(input_str, options, ops[d['answer']])



class GSM8K(SingleSelectAnsLLMEval):
    """
    GSM8K is a dataset class that loads mathematical questions and answers from 
    the Hugging Face datasets (main test set). The dataset is a collection of 
    8.5K high-quality, linguistically diverse grade school math word problems.
    GSM8K dataset is loaded from huggingface datasets: /gsm8k (test set).
    
    Reference: https://huggingface.co/datasets/gsm8k/viewer/main/test
    
    Example data format:
    [{'question': "A robe takes 2 bolts of blue fiber and half that much white 
    fiber. How many bolts in total does it take?", 'answer': 
    "It takes 2/2=<<2/2=1>>1 bolt of white fiber So the total amount of fabric 
    is 2+1=<<2+1=3>>3 bolts of fabric #### 3"}, ...]
    """
    def __init__(self, model, tokenizer, test_n = None, dataset_path='data/llm_general/gsm8k', device = 'cuda', seed = 0):  
        super().__init__(model, tokenizer, test_n, device, seed)   
        # gsm8k dataset now is loaded from huggingface datasets: /gsm8k (test set).
        # https://huggingface.co/datasets/gsm8k
        
        # data = load_dataset("gsm8k", "main")["test"]
        data = load_from_disk(os.path.join(dataset_path, 'main'))["test"]
        self.data = []
        for d in data:
            content = d["question"].strip()
            label = d["answer"].split("#### ")[-1]
            self.data.append({"content": content, "label": label})
        self.prompt = 'Please calculate the following arithmetic questions and choose the correct answer:\n'
        self.shuffle_and_select_data()
    
    def dataset_name(self):
        return 'GSM8K'
    
    def get_input_option(self, idx:int):
        def non_0_normal_int(std):
            std = max(4, abs(std))
            while True:
                t = int(self.rng.normal(0, std))
                if t != 0:
                    return t
        ops = ['A','B','C','D']
        d = self.data[idx]
        input_str = self.prompt + d['content'] + '\n'
        ans_type = float if '.' in d['label'] else int
        ans_number = ans_type(d['label'].replace(',', '')) 
        choices = [non_0_normal_int(ans_number/2) for i in range(3)]
        choices.append(0)
        self.rng.shuffle(choices)
        ans = ops[choices.index(0)]
        choices = [str(c + ans_number) for c in choices]
        input_str += '\n'.join(['('+ops[i]+') '+ c for i, c in enumerate(choices)]) + '\nAnswer: ('
        return self.input_ops_ans_str2ids(input_str, ops, ans)

       
class CSQA(SingleSelectAnsLLMEval):
    """
    CSQA is a dataset class that loads questions and answers from the CommonsenseQA dataset. The dataset is a challenging commonsense question-answering dataset. It comprises 12,247 questions with 5 multiple-choice answers each. 
    CSQA dataset now is loaded from huggingface datasets: /commonsense_qa (val set).
    
    Reference: https://huggingface.co/datasets/commonsense_qa/viewer/default/validation
    
    Example data format:
    ['id': "1afa02df02c908a558b4036e80242fac", 'question': "A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?", 'question_concept': "revolving door", 'choices': { "label": [ "A", "B", "C", "D", "E" ], "text": [ "bank", "library", "department store", "mall", "new york" ] }, 'answerKey': "A" ]
    
    """
    def __init__(self, model, tokenizer, test_n = None, dataset_path='data/llm_general/commonsense_qa', device = 'cuda', seed = 0):  
        super().__init__(model, tokenizer, test_n, device, seed)   
        data = load_from_disk(dataset_path)["validation"]
        self.data = [d for d in data]
        self.prompt = 'Please select the correct answer for the following questions:\n'
        self.shuffle_and_select_data()

    def dataset_name(self):
        return 'CSQA'
    
    def get_input_option(self, idx:int):
        d = self.data[idx]
        input_str = self.prompt + d['question'] + '\n'
        input_str += '\n'.join(['('+l+') '+t for l,t in zip(d['choices']['label'], d['choices']['text'])]) + '\nAnswer: ('
        return self.input_ops_ans_str2ids(input_str, d['choices']['label'], d['answerKey'])




class ANLI(SingleSelectAnsLLMEval):
    """
    The Adversarial Natural Language Inference (ANLI) is a new large-scale NLI 
    benchmark dataset, The dataset is collected via an iterative, adversarial 
    human-and-model-in-the-loop procedure. ANLI is much more difficult than its 
    predecessors including SNLI and MNLI. It contains three rounds. Each round 
    has train/dev/test splits.

    Reference: https://huggingface.co/datasets/facebook/anli
    """
    def __init__(self, model, tokenizer, test_n = None, dataset_path='data/llm_general/anli', device = 'cuda', seed = 0):  
        super().__init__(model, tokenizer, test_n, device, seed)   
        data = load_from_disk(dataset_path)
        self.data = [d for t in ['test_r1', 'test_r2', 'test_r3'] for d in data[t]]
        self.shuffle_and_select_data()

    def dataset_name(self):
        return 'ANLI'
    
    def get_input_option(self, idx:int):
        d = self.data[idx]
        input_str = 'premise: ' + d['premise'] + '\n'
        input_str += 'hypothesis: ' + d['hypothesis'] + '\n'
        # input_str += '(A) entailment\n(B) neutral\n(C) contradiction\n'
        input_str += 'Please determine whether the logical relationship between the '
        input_str += 'hypothesis and the premise is entailment, neutral, or contradiction: '
        ops = ['entailment', 'neutral', 'contradiction']
        ans = ops[d['label']]
        return self.input_ops_ans_str2ids(input_str, ops, ans)



class HellaSwag(SingleSelectAnsLLMEval):
    """
    Reference: https://huggingface.co/datasets/Rowan/hellaswag
    """
    def __init__(self, model, tokenizer, test_n = 1000, dataset_path='data/llm_general/hellaswag', 
                 device = 'cuda', seed = 0):  
        super().__init__(model, tokenizer, test_n, device, seed)   
        data = load_from_disk(dataset_path)
        self.data = [d for d in data['train']]
        self.shuffle_and_select_data()

    def dataset_name(self):
        return 'HellaSwag'
    
    def get_input_option(self, idx:int):
        ops = ['A','B','C','D']
        d = self.data[idx]
        input_str = 'Please select the most likely subsequent scenario in the real world based on the following context description: \n\n'
        input_str += d['ctx'] + '\n\n'
        input_str += ''.join(['('+ops[i]+') ' + o + '\n' for i, o in 
                                enumerate(d['endings'])])
        input_str += '\nAnswer: (' 
        ans = ops[int(d['label'])]
        return self.input_ops_ans_str2ids(input_str, ops, ans)


class COPA(SingleSelectAnsLLMEval):
    """
    Reference: https://huggingface.co/datasets/super_glue
    """
    def __init__(self, model, tokenizer, test_n = None, 
                 dataset_path='data/llm_general/super_glue/copa', 
                 device = 'cuda', seed = 0):  
        super().__init__(model, tokenizer, test_n, device, seed)   
        data = load_from_disk(dataset_path)
        self.data = [d for i in ['train', 'validation'] for d in data[i]]
        self.shuffle_and_select_data()

    def dataset_name(self):
        return 'COPA'
    
    def get_input_option(self, idx:int):
        d = self.data[idx]
        if d['question'] == 'effect':
            input_str = 'Based on the following cause, choose the most likely consequence:'
        elif d['question'] == 'cause':
            input_str = 'Based on the following result, choose the most likely cause:'
        else:
            raise
        input_str += '\n\n' + d['premise'] + '\n'
        input_str += '(A) ' + d['choice1'] + '\n'
        input_str += '(B) ' + d['choice2'] + '\n\n'
        input_str += 'Answer: (' 
        ops = ['A','B']
        ans = ops[d['label']]
        return self.input_ops_ans_str2ids(input_str, ops, ans)
 