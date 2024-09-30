from editors.editor import BaseEditor, EditorConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple 
from dataclasses import dataclass, asdict
import numpy as np
from copy import deepcopy
import torch, os, yaml
from torch.utils.tensorboard import SummaryWriter 
from torch.optim import Adam
from .models import KnowledgeRepModel, PromptTransformer
from utils.data import ParallelDataset
from datetime import datetime
from tqdm import tqdm
from torch import nn

@dataclass
class RECIPEConfig(EditorConfig):
    @dataclass
    class TrainingConfig():
        krm_lr: float
        pt_lr: float
        relia_lambda: float
        gen_lambda: float
        loc_lambda: float
        contra_lambda: float
        query_knowledge_t: float
        query_prototype_t: float
        constra_hinge_scale: float # w/hinge >= 1, w/o hinge== 999999 
        edit_hinge_scale: float # w/hinge >= 1, w/o hinge== 999999 
        # set in train_init
        batch_size:int = None
        sample_count:int = None
        random_seed:int = None
        eps:float = 1e-8
         
    prompt_token_n: int
    edit_model_name: str
    knowledge_rep_dim: int
    knowl_rep_prot_token_n: int
    model_hidden_size: int
    begin_layer_path:str
    lm_head_path:str
    training: TrainingConfig

    @classmethod
    def from_yaml(self, fpath):
        with open(fpath, "r") as f:
            data = yaml.safe_load(f)
        data['training'] = self.TrainingConfig(**data['training'])
        return self(**data)
    @classmethod
    def from_json(self, fpath):
        raise
    
class RECIPE(BaseEditor):
    def __init__(self, 
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: RECIPEConfig,
        device = 'cuda',
        krm_base_path = 'models/roberta-base',
        retr_top_k = 1,
        retr_min_sim = -999,
        auto_retrieve = True, 
        ckpt_path = None 
    ):
        super().__init__(model, tokenizer, device)
        self.cfg = config
        # initialize model & parameters
        self.knowl_rep_model = KnowledgeRepModel(config.knowledge_rep_dim, 
                    config.knowl_rep_prot_token_n, self.device, krm_base_path)
        self.prompt_transformer = PromptTransformer(config.knowledge_rep_dim, 
            config.model_hidden_size, config.prompt_token_n, self.device)
        # initialize hooks
        self.begin_layer = find_module(self.model, self.cfg.begin_layer_path)
        self.lm_head = find_module(self.model, self.cfg.lm_head_path)
        self.model.forward = self.register_model_forward_hook(self.model, 
                        self.model.forward, self.begin_layer, self.lm_head)
        self.begin_layer_hook = self.register_editing_hook(self.begin_layer, self.lm_head)
        # initialize editing prompts
        self.restore_to_original_model()
        self.auto_retrieve = auto_retrieve 
        self.retr_top_k = retr_top_k
        self.retr_min_sim = retr_min_sim
        self.set_train(False) 
        if ckpt_path != None:
            self.load_ckpt(ckpt_path, load_opt = False)
    
    ############################################################################
    ############################# Initialize ###################################
    ############################################################################
    def register_editing_hook(self, begin_layer, lm_head_layer):
        def forward_pre_hook(module, args):
            # If do not has past_key_values, add editing prompts before reps.
            if not module.has_past_kv:
                args = args[0]
                args = torch.stack([
                    torch.cat([p, inp[:-len(p) if len(p) != 0 else None]], 0)
                    for inp, p in zip(args, self.adopted_prompts)], 0)
                return (args, )
        def forward_hook(module, args, output):
            if not module.has_past_kv:
                max_n = max([len(p) for p in self.adopted_prompts])
                output = torch.stack([
                    ot[len(p):len(p)-max_n if len(p)-max_n != 0 else None]
                    for ot, p in zip(output, self.adopted_prompts)], 0)
            return output
        begin_layer_hook = begin_layer.register_forward_pre_hook(forward_pre_hook)
        lm_head_layer_hook = lm_head_layer.register_forward_hook(forward_hook)
        return [begin_layer_hook, lm_head_layer_hook]

    def register_model_forward_hook(self, model, model_forward, begin_layer, lm_head):
        if hasattr(model, 'recipe_hooked'):
            return model_forward
        model.recipe_hooked = True
        def forward_recipe(**kargs):
            if 'past_key_values' in kargs and kargs['past_key_values'] != None:
                begin_layer.has_past_kv = True
                lm_head.has_past_kv = True
            else:
                begin_layer.has_past_kv = False
                lm_head.has_past_kv = False
                b, l = kargs['input_ids'].shape
                inp_sents = [self.tokenizer.decode(i, skip_special_tokens=True) 
                             for i in kargs['input_ids']]
                if self.auto_retrieve: 
                    retrieved_ids = self.retrieve_and_get_ids_sim(inp_sents)[0]
                    # print(retrieved_ids)
                    self.adopted_prompts = [self.prompts_base[i].reshape(
                        len(i)*self.cfg.prompt_token_n, self.cfg.model_hidden_size) 
                        for i in retrieved_ids]
                if len(self.adopted_prompts) != b:
                    print(len(self.adopted_prompts), b) 
                    raise ValueError
                pad = torch.ones([b, max([len(i) for i in self.adopted_prompts])], 
                                 dtype = torch.long).to(self.device)
                if 'attention_mask' in kargs and kargs['attention_mask'] != None:
                    kargs['attention_mask'] = torch.cat([kargs['attention_mask'], pad], 1)
                kargs['input_ids'] = torch.cat([kargs['input_ids'], pad * self.tokenizer.pad_token_id], 1)
            return model_forward(**kargs)
        return forward_recipe

    ############################################################################
    ############################# RECIPE Edit Related ############################
    ############################################################################
    def retrieve_and_get_ids_sim(self, input_queries:List[str]):
        query_reps = self.knowl_rep_model(input_queries, knowl_or_query = 'q') # [n, knowledge_rep_dim] 
        sim_matrx = (query_reps @ self.knowledge_base.T) / self.cfg.knowledge_rep_dim**0.5 # cross_cos_sim(query_reps, self.knowledge_base) # [n, edit_n]
        sim_with_prototype = sim_matrx[:, :1] 
        sorted_sim, order = torch.sort(sim_matrx, 1, True) # [n, edit_n]
        mask = sorted_sim[:, :self.retr_top_k] > self.retr_min_sim
        mask &= sorted_sim[:, :self.retr_top_k] > sim_with_prototype
        retrieved_ids = torch.masked_select(order[:, :self.retr_top_k], mask)
        retrieved_ids = torch.split(retrieved_ids, mask.sum(1).tolist()) # retrieved indexes
        return retrieved_ids, (sorted_sim, order) # [retr_ids_1, retr_ids_2, ..., retr_ids_n]

    ############################################################################
    ############################# Editor Basic Functions #######################
    ############################################################################
    def name_of_editor_and_model(self)->Tuple[str, str]:
        return 'recipe', self.cfg.edit_model_name

    def if_can_batch_edit(self):
        return True
 
    def restore_to_original_model(self):
        self.knowledge_base_nl = ['<Knowledge_Representation_Prototype>'] # [edit_n, knowledge_rep_dim]
        self.knowledge_base = self.knowl_rep_model.get_knowl_rep_prot() # [edit_n, knowledge_rep_dim]
        self.prompts_base = torch.zeros([1, self.cfg.prompt_token_n, self.cfg.model_hidden_size], 
            device = self.device) # [edit_n, prompt_token_n, model_hidden_size]
        self.adopted_prompts = [] # List[torch.Tensor], len(List) = btach_size, Tensor.size = [retr_n * prompt_token_n, model_hidden_size]

    def edit_batch(self, requests: List[Dict]):
        '''requests = [
            {'prompt':str, 'subject':str, 'target_new':str}
            {'prompt':str, 'subject':str, 'target_new':str}, ...
        ]
        '''
        rs = []
        for r in requests:
            if r['prompt'][-1] != ' ' and r['target_new'][0] != ' ':
                rs.append(r['prompt'] + ' ' + r['target_new'])
            else:
                rs.append(r['prompt'] + r['target_new'])
        self.knowledge_base_nl.extend(rs)
        new_reps = self.knowl_rep_model(rs, knowl_or_query = 'k')
        new_prompts = self.prompt_transformer(new_reps)
        self.knowledge_base = torch.cat([self.knowledge_base, new_reps], 0)
        self.prompts_base = torch.cat([self.prompts_base, new_prompts], 0)

    def edit_one_piece(self, request: Dict) -> None:
        """
        request = {'prompt':str, 'subject':str, 'target_new':str}
        """
        self.edit_batch([request])

    ############################################################################
    ############################# RECIPE Training ################################
    ############################################################################
    def set_train(self, if_train = False):
        self.model.train(False)
        self.model.requires_grad_(False)
        self.knowl_rep_model.train(if_train)
        self.knowl_rep_model.requires_grad_(if_train)
        self.prompt_transformer.train(if_train)
        self.prompt_transformer.requires_grad_(if_train)
        self.auto_retrieve = not if_train

    def train_init(self, sample_count, get_data_by_ids, batch_size, 
            records_dir:str = 'train_records', train_name_prefix = None, 
            train_name:str = None, load_ckpt_path:str = None, 
            save_ckpt_per_i = 3000, log_per_i = 10, random_seed = None):  
        '''
        Used to initialize `ParallelDataset`:
            sample_count: count of used data in dataset.
            get_data_by_ids: function getting data by ids, assume data structure: (
                batch_knowledge: List[str], len = batch_size
                contra_q: List[str], len = batch_size * 3
                contra_sim_m: torch.Tensor[batch_size, batch_size * 3]
                batch_relia_xym: (input_ids, label_ids, masks), 
                batch_gen_xym: {
                    loss_name_1: (input_ids, label_ids, masks),
                    loss_name_2: (input_ids, label_ids, masks), ...
                },
                batch_loc_xym: {
                    loss_name_1: (input_ids, masks)
                    loss_name_2: (input_ids, masks), ...
                }  
            ), where `input_ids`, `label_ids`, and `label_ids` are with shape 
            [batch_size, length]
        '''
        # initialize data generator
        self.rng = np.random.default_rng(random_seed)
        self.data_generator = ParallelDataset(sample_count, get_data_by_ids, 
            batch_size, True, 16, False, random_seed)
        # initialize checkpoint/log directory and writer
        t = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
        train_name = (train_name_prefix + '-' if train_name_prefix else "") + \
            (train_name if train_name else t)
        records_dir = os.path.join(records_dir, *self.name_of_editor_and_model(), train_name)
        self.save_ckpt_dir = os.path.join(records_dir, 'checkpoints')
        if not os.path.exists(self.save_ckpt_dir):
            os.makedirs(self.save_ckpt_dir)
        logs_path = os.path.join(records_dir, 'logs')
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        with open(os.path.join(records_dir, 'config.yaml'), 'w') as f:
            self.cfg.training.batch_size = batch_size
            self.cfg.training.sample_count = sample_count
            self.cfg.training.random_seed = random_seed
            yaml.dump(asdict(self.cfg), f)
        self.log_writer = SummaryWriter(logs_path)
        self.save_ckpt_per_i = save_ckpt_per_i
        self.log_per_i = log_per_i
        # initialize optimizer and load checkpoints
        self.opt = Adam([
            {'params': self.knowl_rep_model.parameters(), 'lr': self.cfg.training.krm_lr},
            {'params': self.prompt_transformer.parameters(), 'lr': self.cfg.training.krm_lr}])
        if load_ckpt_path and os.path.isfile(load_ckpt_path):
            self.ema_loss = self.load_ckpt(load_ckpt_path, True)  
        else:
            self.train_i, self.train_epoch = 1, 1
            self.ema_loss = 1

    def train(self, epochs):
        if self.log_writer == None:
            raise "Call `self.train_init()` to initialize training first!"
        print('Checkpoints dir: ', self.save_ckpt_dir)
        start_epoch = self.train_epoch
        self.set_train(True) 
        for self.train_epoch in range(start_epoch, epochs + 1): 
            progress_bar = tqdm(total = self.data_generator.sample_count, 
                position = 0, leave = True, desc = "Epoch %d"%self.train_epoch, dynamic_ncols = True)
            for contra_data, edit_data in self.data_generator:
                # train after edit
                log_dict = self.__train_a_batch__(*contra_data, *edit_data)
                # log
                log_dict['Epoch'] = self.train_epoch
                if self.train_i % self.log_per_i == 0:
                    self.write_logs(self.train_i, log_dict)
                if self.train_i % self.save_ckpt_per_i == 0:
                    self.save_ckpt(self.train_i, self.train_epoch, log_dict['Loss'])
                self.train_i += 1 
                progress_bar.update(len(contra_data[0]))
            progress_bar.close() 
        self.set_train(False)

    def __train_a_batch__(self, contra_knowl, contra_q_rel, contra_q_gen, contra_q_loc,
                          batch_relia_xym, batch_gen_xym, batch_loc_xym):
        # prediction before edit for locality loss
        with torch.no_grad():
            for loss_name, sp in batch_loc_xym.items():
                input_ids, _, masks = sp
                self.adopted_prompts = [torch.zeros([0, self.cfg.model_hidden_size], device = self.device)] * len(input_ids)
                pre_logits = self.model(input_ids = input_ids).logits
                batch_loc_xym[loss_name] = ((input_ids, masks), pre_logits)
        loss = 0 
        bsz = len(contra_knowl)
        eps = self.cfg.training.eps
        cc = self.rng.choice([0, 1], bsz)
        q1 = [[contra_q_rel, contra_q_gen][c][i] for i, c in enumerate(cc)]
        q2 = [[contra_q_rel, contra_q_gen][c][i] for i, c in enumerate(1-cc)]
        q1_reps = self.knowl_rep_model(q1, knowl_or_query = 'q') # [bsz, rep_dim]
        q2_reps = self.knowl_rep_model(q2, knowl_or_query = 'q') # [bsz, rep_dim]
        knowledge_reps = self.knowl_rep_model(contra_knowl, knowl_or_query = 'k') # [bsz, rep_dim]
        knowl_rep_prot = self.knowl_rep_model.get_knowl_rep_prot() 
        knowl_reps_with_proto = torch.cat([knowledge_reps, knowl_rep_prot])
        scale_factor = 1 / self.cfg.knowledge_rep_dim**0.5
        chs = self.cfg.training.constra_hinge_scale
        ehs = self.cfg.training.edit_hinge_scale
        # reliability/generality loss_contra_q1
        sim_q1 = (q1_reps @ knowl_reps_with_proto.T) * scale_factor # [bsz, bsz+1]
        sim_q1 = torch.softmax(sim_q1 * self.cfg.training.query_knowledge_t, 1)
        loss_contra_q1 = - torch.log(torch.diag(sim_q1) + eps).mean(0)
        # reliability/generality loss_contra_q2
        sim_q2 = (q2_reps @ knowledge_reps.T) * scale_factor # [bsz, bsz]
        sim_q2 = sim_q2 * (1 - torch.eye(bsz, device=self.device))
        sim_q2 = sim_q2 + torch.diag((q2_reps @ knowl_rep_prot.T)[:, 0] * scale_factor)
        sim_q2 = torch.softmax(sim_q2 * self.cfg.training.query_prototype_t, 1)
        second_sim_q2 = torch.topk(sim_q2, 2, 1).values[:, 1]
        sim_q2 = torch.diag(sim_q2) 
        sim_q2 = torch.masked_select(sim_q2, sim_q2 < second_sim_q2 * chs)
        if len(sim_q2) == 0:
            loss_contra_q2 = 0
        else:
            loss_contra_q2 = - torch.log(sim_q2 + eps).mean(0) 
        # locality loss_contra_q3 
        losses_contra_q3 = {}
        loss_contra_q3 = 0
        for k, cql in contra_q_loc.items():
            q3_reps = self.knowl_rep_model(cql, knowl_or_query = 'q') # [bsz, rep_dim]
            sim_q3 = (q3_reps @ knowl_reps_with_proto.T) * scale_factor # [bsz, bsz+1]
            sim_q3 = torch.softmax(sim_q3 * self.cfg.training.query_prototype_t, 1)
            second_sim_q3 = torch.topk(sim_q3, 2, 1).values[:, 1]
            sim_q3 = sim_q3[:, -1]
            sim_q3 = torch.masked_select(sim_q3, sim_q3 < second_sim_q3 * chs)
            if len(sim_q3) == 0:
                l = 0
            else:
                l = - torch.log(sim_q3 + eps).mean(0) 
            losses_contra_q3[k] = l
            loss_contra_q3 += l
        # sum
        loss_contra = loss_contra_q1 + loss_contra_q2 + loss_contra_q3
        loss += loss_contra * self.cfg.training.contra_lambda
        # edit loss
        self.adopted_prompts = [i for i in self.prompt_transformer(knowledge_reps)]
        # compute reliability loss
        (input_ids, label_ids, masks) = batch_relia_xym
        relia_loss = hinge_label_loss(self.model, input_ids, label_ids, masks, True, eps, ehs)
        loss += relia_loss * self.cfg.training.relia_lambda
        # compute generality loss
        gen_losses = {}
        for loss_name, sp in batch_gen_xym.items():
            input_ids, label_ids, masks = sp
            gen_loss = hinge_label_loss(self.model, input_ids, label_ids, masks, True, eps, ehs)
            gen_losses[loss_name] = gen_loss
            loss += gen_loss * self.cfg.training.gen_lambda
        # compute locality loss
        loc_losses = {}
        for loss_name, sp in batch_loc_xym.items():
            (input_ids, masks), pre_logits = sp
            post_logits = self.model(input_ids = input_ids).logits
            loc_loss = logit_KL_loss(pre_logits, post_logits, masks)
            loc_losses[loss_name] = loc_loss
            loss += loc_loss * self.cfg.training.loc_lambda
        # update
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        self.ema_loss = self.ema_loss + (loss.detach() - self.ema_loss) / self.log_per_i
        log_dict = {
            'Loss': loss,
            'EMA Loss': self.ema_loss, 
            'Contrastive loss': loss_contra,
            'Contrastive loss q2k': loss_contra_q1,
            'Contrastive loss q2prot': loss_contra_q2 + loss_contra_q3,
            'Contrastive loss q2prot-rg': loss_contra_q2,
            'Contrastive loss q2prot-loc': losses_contra_q3,
            'Reliability loss': relia_loss,
            'Generality loss': gen_losses,
            'Locality loss': loc_losses
        }
        return log_dict

    def write_logs(self, i, logs:dict):
        for log_name, log in logs.items():
            if type(log) == dict:
                logs1 = {}
                for n, l in log.items():
                    logs1[log_name + '-' + n] = l
                self.write_logs(i, logs1)
            else:   
                self.log_writer.add_scalar(log_name, log, i)

    def save_ckpt(self, i:int, epoch:int, loss:float):
        ckpt_name = 'epoch-%d-i-%d-ema_loss-%.4f'%(epoch, i, self.ema_loss)
        ckpt_path = os.path.join(self.save_ckpt_dir, ckpt_name)
        ckpt = {
            'i': i,
            'epoch': epoch,
            'loss': loss,
            'knowl_rep_model': self.knowl_rep_model.state_dict(),
            'prompt_transformer': self.prompt_transformer.state_dict(),
            'opt': self.opt.state_dict()
        }
        torch.save(ckpt, ckpt_path)

    def load_ckpt(self, ckpt_path, restrict = True, load_opt = True):
        ckpt = torch.load(ckpt_path, 'cpu')
        self.train_i = ckpt['i']
        self.train_epoch = ckpt['epoch']
        self.knowl_rep_model.load_state_dict(ckpt['knowl_rep_model'], restrict)
        self.prompt_transformer.load_state_dict(ckpt['prompt_transformer'], restrict)
        if load_opt:
            self.opt.load_state_dict(ckpt['opt'])
        print('Load RECIPE checkpoints from', ckpt_path)
        return ckpt['loss']


def hinge_label_loss(model, input_ids:torch.Tensor, label_ids:torch.Tensor, 
            masks:torch.Tensor, average = True, eps = 1e-8, hinge_scale = 1.1):
    # input_ids/label_ids/masks: [batch, max_len]
    logits = model(input_ids = input_ids).logits # [batch, max_len, voc_size]
    pre_p = torch.softmax(logits, 2) # [batch, max_len, voc_size]
    second_pre_p = torch.topk(pre_p, 2, -1).values[:, :, 1] # [batch, max_len]
    pre_p = pre_p.gather(-1, label_ids.unsqueeze(-1)).squeeze(-1) # [batch, max_len]
    masks = masks * (pre_p < second_pre_p * hinge_scale)
    pre_p = torch.masked_select(pre_p, masks.to(bool))
    loss = - torch.log(pre_p + eps).sum() # [batch, max_len] 
    if average:
        sm = masks.sum() 
        if sm != 0:
            loss = loss / sm
    return loss
 
def label_loss(model, input_ids:torch.Tensor, label_ids:torch.Tensor, masks:torch.Tensor, average = True):
    # input_ids/label_ids/masks: [batch, max_len]
    logits = model(input_ids = input_ids).logits
    log_pre_p = torch.log_softmax(logits, 2) # [batch, max_len, voc_size]
    log_pre_p = log_pre_p.gather(-1, label_ids.unsqueeze(-1)).squeeze(-1) # [batch, max_len]
    loss = -(log_pre_p * masks).sum()
    if average:
        loss = loss / masks.sum() 
    return loss

def logit_KL_loss(logits1:torch.Tensor, logits2:torch.Tensor, masks:torch.Tensor, average = True):
    # logits1/logits2: [batch, max_len, voc_size], masks: [batch, max_len]
    log_p1 = torch.log_softmax(logits1, 2)
    log_p2 = torch.log_softmax(logits2, 2)
    p1 = torch.softmax(logits1, 2)
    kl_loss = (p1 * (log_p1 - log_p2)).sum(2) # [batch, max_len]
    loss = (kl_loss * masks).sum()
    if average:
        loss = loss / masks.sum() 
    return loss


def find_module(module, module_path:str):
    for comp in module_path.split('.'):
        if hasattr(module, comp):
            module = getattr(module, comp)
        elif comp.isdigit():
            module = module[int(comp)]
        else:
            raise RuntimeError(f"Couldn't find child module {comp}")
    return module

def cross_cos_sim(a, b):
    # compute cos similarly: [n, d], [m, d] -> [n, m]
    a = torch.nn.functional.normalize(a, 2, 1)
    b = torch.nn.functional.normalize(b, 2, 1)
    return a @ b.T

def get_mask_matrix(v, max_n = None):
    max_n = max_n if max_n != None else torch.max(v)
    return torch.arange(1, max_n + 1).unsqueeze(0) <= v.unsqueeze(1)
