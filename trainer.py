import os
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer, AutoModelForCausalLM, DataCollatorWithPadding, DefaultDataCollator, pipeline, AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model
import pdb
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from trl import RewardTrainer
from torch.distributions import Categorical
from npy_append_array import NpyAppendArray
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

def logsumexp(x):
    c = x.max()
    return c + torch.log(torch.sum(torch.exp(x-c)))

class CustomTrainer(Trainer):
    def __init__(self, pretrained_model_path, scaling, invert, reg_type, opt_type, pr_scale, model_name, dataset, 
                 higheroutcomebetter, outcome_model_path, num_labels, results_dir, eightbit, paired, metric, 
                 c_ipw, c_rlhf, c_entropy, norm_rlhf_term, stabilize, norm_ipw_term, use_pr_est, model_path, save,
                 #  outcome_tokenizer_path,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        load_in_8bit = False
        self.paired = False
        if eightbit:
            load_in_8bit = True
        if paired:
            self.paired = True
        # self.pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_path, load_in_8bit=load_in_8bit, device_map='auto')
        # self.pretrained_model.config.pad_token_id = self.pretrained_model.config.bos_token_id
        # self.pretrained_model.eval()
        # peft_config = LoraConfig(
        # task_type=TaskType.CAUSAL_LM, 
        #     inference_mode=True, 
        #     r=8, 
        #     lora_alpha=32, 
        #     lora_dropout=0.1)

        # self.pretrained_model = get_peft_model(self.pretrained_model, peft_config)
        # self.pretrained_model.to(DEVICE)

        self.scaling = scaling
        self.invert = invert
        self.reg_type = reg_type
        self.opt_type = opt_type
        self.pr_scale = pr_scale
        self.model_name = model_name
        self.dataset = dataset
        self.higheroutcomebetter = higheroutcomebetter
        self.metric = metric
        self.c_ipw = c_ipw
        self.c_rlhf = c_rlhf
        self.c_entropy = c_entropy
        self.results_dir = results_dir
        self.norm_file = os.path.join(self.results_dir, 'models/{}/{}.txt'.format(self.dataset, self.model_name))
        self.norm_log_probs = None
        self.norm_rlhf_term = False
        if norm_rlhf_term:
            self.norm_rlhf_term = True
        self.stabilize = False
        if stabilize:
            self.stabilize = True
        self.norm_ipw_term = False
        if norm_ipw_term:
            self.norm_ipw_term = True
        self.use_pr_est = False
        if use_pr_est:
            self.use_pr_est = True
        self.model_path = model_path
        self.save = False
        if save:
            self.save = True

        if (self.opt_type == 'exact' or self.opt_type == 'bias_corrected_rlhf' or self.opt_type == 'sanity_check') and os.path.exists(self.norm_file):
            print('Loading norm file')
            with open(self.norm_file, 'r') as f:
                content = f.read()
                try:
                    self.norm_log_probs = float(content.strip())
                except ValueError:
                    print('The file does not contain a valid float')
        # if outcome_model_path is not None:
        #     self.outcome_model = AutoModelForSequenceClassification.from_pretrained(outcome_model_path, load_in_8bit=load_in_8bit, device_map='auto', num_labels=num_labels)
        #     self.outcome_model.config.pad_token_id = self.outcome_model.config.bos_token_id
            # self.outcome_model = get_peft_model(self.outcome_model, peft_config)
            # self.outcome_model.to(DEVICE)
        self.num_labels = num_labels
            # self.outcome_model = pipeline(
            #     'text-classification', 
            #     model=outcome_model_path, 
            #     tokenizer=outcome_tokenizer_path,
            #     device=0)
            
    def compute_loss(self, model, inputs, return_outputs=False):

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        raw_text = inputs["raw_text"]
        outcome = inputs["outcome"]
        pretrained_logits = inputs["pr_sentence_log_probs"]
        pred_outcome = inputs["pred_outcome"]
        if self.paired:
            pred_outcome_2 = inputs["pred_outcome_2"]

        if self.invert:
            if 'hk' in self.dataset:
                outcome = 100 - outcome
                pred_outcome = 100 - pred_outcome
                if self.paired:
                    pred_outcome_2 = 100 - pred_outcome_2
            elif self.dataset == 'hatespeech' or self.dataset == 'emobank_binary':
                outcome = -outcome
                pred_outcome = -pred_outcome
                if self.paired:
                    pred_outcome_2 = -pred_outcome_2
            elif self.dataset == 'emobank':
                outcome = 6 - outcome
                pred_outcome = 6 - pred_outcome
                if self.paired:
                    pred_outcome_2 = 6 - pred_outcome_2

        if self.paired:
            input_ids_2 = inputs['input_ids_2']
            attention_mask_2 = inputs['attention_mask_2']
            pretrained_logits_2 = inputs['pr_sentence_log_probs_2']
            
            outputs_2 = model(input_ids=input_ids_2, attention_mask=attention_mask_2, output_hidden_states=True)
            logits_2 = outputs_2.logits
                
            # pretrained_outputs_2 = self.pretrained_model(input_ids=input_ids_2, attention_mask=attention_mask_2, output_hidden_states=True)
            # pretrained_logits_2 = pretrained_outputs_2.logits.detach()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        logits = outputs.logits
        
        finetuned_embeddings = outputs.hidden_states[-1]
        # pretrained_outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # pretrained_embeddings = pretrained_outputs.hidden_states[-1].detach()
        # pretrained_logits = pretrained_outputs.logits.detach()

        # if self.opt_type == 'sanity_check':
        #     loss = self.sanity_check_loss(logits, input_ids, outcome, pred_outcome)
        if self.opt_type == 'pr_ratio':
            loss = self.pr_ratio_loss(logits, pretrained_logits, input_ids, outcome, self.invert)
        elif self.opt_type == 'exact':
            loss = self.exact_loss(logits, pretrained_logits, input_ids, outcome, self.invert)
        elif self.opt_type == 'ce':
            loss = self.ce_loss(logits, input_ids, outcome, self.invert) # Negative log prob, kind of
        elif self.opt_type == 'ce_diff':     
            loss = self.ce_diff_loss(logits, pretrained_logits, input_ids, outcome, self.invert)
        elif (self.opt_type == 'bias_corrected_rlhf') or (self.opt_type == 'sanity_check'):
            # outcome_output = self.outcome_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # outcome_logits = outcome_output.logits.detach()
            # if self.num_labels == 1:
            #     pred_outcome = outcome_logits.squeeze()
            # elif self.num_labels == 2:
            #     pred_probs = F.softmax(outcome_logits, dim=-1)
            #     idx = outcome.clone()
            #     idx[idx == -1] = 0
            #     pred_probs = pred_probs.gather(dim=1, index=idx.view(-1, 1)).squeeze()
            #     pred_outcome = pred_probs*outcome
            
            if self.opt_type == 'sanity_check':
                loss = self.sanity_check_loss(logits, input_ids, outcome, pred_outcome, self.metric)
            else:
                if self.paired:
                    # outcome_output_2 = self.outcome_model(input_ids=input_ids_2, attention_mask=attention_mask_2, output_hidden_states=True)
                    # outcome_logits_2 = outcome_output_2.logits.detach()
                    # if self.num_labels == 1:
                    #     pred_outcome_2 = outcome_logits_2.squeeze()
                    # elif self.num_labels == 2:
                    #     pred_probs_2 = F.softmax(outcome_logits_2, dim=-1)
                    #     pred_probs_2 = pred_probs_2.gather(dim=1, index=idx.view(-1, 1)).squeeze()
                    #     pred_outcome_2 = pred_probs_2*outcome
                    loss = self.bias_corrected_rlhf_loss(logits, pretrained_logits, input_ids, outcome, pred_outcome, self.invert, self.paired, input_ids_2, pred_outcome_2, logits_2, pretrained_logits_2)
                else:
                    loss = self.bias_corrected_rlhf_loss(logits, pretrained_logits, input_ids, outcome, pred_outcome, self.invert)

        reg = 0
        if self.reg_type == 'sim':
            reg = self.similarity_loss(finetuned_embeddings, pretrained_embeddings) # Cosine similarity regularization
        elif self.reg_type == 'kl':
            reg = self.kl_loss(logits, pretrained_logits) # Look into why this gets funky with llama-2

        # pdb.set_trace()
        # total_loss = scaling*reg - loss
        # total_loss = reg - scaling*loss
        # total_loss = scaling*reg
        total_loss = loss + self.scaling*reg
        # total_loss = loss

        if return_outputs:
            return (total_loss, outputs)
        
        return total_loss

    def optimal_loss(self, logits, input_ids, outcome):
        probs = F.softmax(logits, dim=-1)
        token_log_probs = torch.log(torch.gather(probs, dim=2, index=input_ids.unsqueeze(-1)))
        token_log_probs = token_log_probs.view(-1, token_log_probs.shape[1])
        sentence_log_probs = torch.sum(token_log_probs, -1)
        loss = torch.mean(sentence_log_probs*outcome)

        # get text probability (log prob?) from "model"
        # treat P^R as fixed for now
        # Multiply ratio by outcome

        return loss
    
    def ce_loss(self, logits, input_ids, outcome, inverted=False):
        shift_labels = input_ids[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()
        # Calculate per-token loss
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none') # CLM objective
        # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1), reduction='none')
        loss_per_sample = loss.view(shift_logits.size(0), -1).mean(axis=1)
        if inverted:
            # loss_scaled = (loss_per_sample * outcome).mean()
            loss_scaled = (loss_per_sample * outcome / 100).mean()
        else:
            loss_scaled = (loss_per_sample / (outcome+1) * 100).mean() # Scaling by outcome
        # loss_scaled = (loss_per_sample * outcome).mean()
        # loss_scaled = loss.view(input_ids.size(0), -1)*outcome.view(-1, 1)

        return loss_scaled
        # return loss.mean()

    def pr_ratio_loss(self, logits, pretrained_logits, input_ids, outcome, inverted=False):
        
        token_log_probs = torch.gather(logits, dim=2, index=input_ids.unsqueeze(-1))
        token_log_probs = token_log_probs.view(-1, token_log_probs.shape[1])
        sentence_log_probs = torch.sum(token_log_probs, -1)

        pr_token_log_probs = torch.gather(pretrained_logits, dim=2, index=input_ids.unsqueeze(-1))
        pr_token_log_probs = pr_token_log_probs.view(-1, pr_token_log_probs.shape[1])
        pr_sentence_log_probs = torch.sum(pr_token_log_probs, -1)

        log_ratio = pr_sentence_log_probs - sentence_log_probs

        loss = (log_ratio - torch.log(outcome+1)).mean()

        # Not exactly sure how to handle this because you want it to decrease with bigger outcome and decrease with bigger sentence_log_probs

        # loss = (log_ratio/(outcome+1)).mean()

        # prob_ratio = torch.exp(log_ratio)

        # loss = (prob_ratio / (outcome+1) * 100).nanmean()

        return loss
    
    def ce_diff_loss(self, logits, pretrained_logits, input_ids, outcome, inverted=False):
        shift_labels = input_ids[..., 1:].contiguous()
        shift_logits = (logits[..., :-1, :] - pretrained_logits[..., :-1, :]).contiguous()
        # Calculate per-token loss
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none') # CLM objective
        # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1), reduction='none')
        loss_per_sample = loss.view(shift_logits.size(0), -1).mean(axis=1)
        if inverted:
            loss_scaled = (loss_per_sample * outcome).mean()
        else:
            # loss_scaled = (loss_per_sample / (outcome+1) * 100).mean() # Scaling by outcome
            loss_scaled = (loss_per_sample / ((outcome+1)/1000)).mean() # Scaling by outcome; boosting influence of outcome
        # loss_scaled = (loss_per_sample * outcome).mean()
        # loss_scaled = loss.view(input_ids.size(0), -1)*outcome.view(-1, 1)

        return loss_scaled
    
    def exact_loss(self, logits, pretrained_logits, input_ids, outcome, inverted=False):
        # max_logits, _ = torch.max(pretrained_logits, dim=-1, keepdim=True)
        # logits = logits - max_logits
        # pretrained_logits = pretrained_logits - max_logits

        token_log_probs = torch.gather(logits, dim=2, index=input_ids.unsqueeze(-1))
        token_log_probs = token_log_probs.view(-1, token_log_probs.shape[1])
        # sentence_log_probs = torch.mean(token_log_probs, -1)
        sentence_log_probs = torch.sum(token_log_probs, -1)

        pr_token_log_probs = torch.gather(pretrained_logits, dim=2, index=input_ids.unsqueeze(-1))
        pr_token_log_probs = pr_token_log_probs.view(-1, pr_token_log_probs.shape[1])
        # pr_sentence_log_probs = torch.mean(pr_token_log_probs, -1)
        pr_sentence_log_probs = torch.sum(pr_token_log_probs, -1)

        if self.norm_log_probs == None:
        # if not os.path.exists(self.norm_file):
            self.norm_log_probs = torch.max(pr_sentence_log_probs)
            with open(self.norm_file, 'w') as f:
                f.write(str(self.norm_log_probs.item()))

        log_diff = sentence_log_probs - pr_sentence_log_probs
        norm_log_diff = pr_sentence_log_probs - self.norm_log_probs
        pr_ratio = torch.exp(log_diff - torch.logsumexp(log_diff, dim=0))
        loss = pr_ratio * outcome / self.pr_scale * torch.exp(norm_log_diff - torch.logsumexp(norm_log_diff, dim=0))
        if self.higheroutcomebetter:
            loss = -loss
        loss = torch.nanmean(loss[loss != torch.inf])
        # print(pr_ratio)
        # print(torch.exp(norm_log_diff))

        return loss
    
    def bias_corrected_rlhf_loss(self, logits, pretrained_logits, input_ids, outcome, pred_outcome, inverted=False, 
                                 paired=False, input_ids_2=None, pred_outcome_2=None, logits_2=None, pretrained_logits_2=None):
        
 
        token_log_probs = torch.gather(logits, dim=2, index=input_ids.unsqueeze(-1))
        token_log_probs = token_log_probs.view(-1, token_log_probs.shape[1])
        sentence_log_probs = torch.sum(token_log_probs, -1)

        # pr_token_log_probs = torch.gather(pretrained_logits, dim=2, index=input_ids.unsqueeze(-1))
        # pr_token_log_probs = pr_token_log_probs.view(-1, pr_token_log_probs.shape[1])
        # pr_sentence_log_probs = torch.sum(pr_token_log_probs, -1)
        pr_sentence_log_probs = pretrained_logits

        if paired:
            token_log_probs_2 = torch.gather(logits_2, dim=2, index=input_ids_2.unsqueeze(-1))
            token_log_probs_2 = token_log_probs_2.view(-1, token_log_probs_2.shape[1])
            sentence_log_probs_2 = torch.sum(token_log_probs_2, -1)

            # pr_token_log_probs_2 = torch.gather(pretrained_logits_2, dim=2, index=input_ids_2.unsqueeze(-1))
            # pr_token_log_probs_2 = pr_token_log_probs_2.view(-1, pr_token_log_probs_2.shape[1])
            # pr_sentence_log_probs_2 = torch.sum(pr_token_log_probs_2, -1)
            pr_sentence_log_probs_2 = pretrained_logits_2

        if self.norm_log_probs == None:
        # if not os.path.exists(self.norm_file):
            self.norm_log_probs = torch.max(pr_sentence_log_probs)
            print('Writing norm file')
            with open(self.norm_file, 'w') as f:
                f.write(str(self.norm_log_probs.item()))

        log_diff = sentence_log_probs - pr_sentence_log_probs
        if self.save:
            log_diff_filename = os.path.join(self.model_path, 'log_diff.npy')
            with NpyAppendArray(log_diff_filename) as npaa:
                npaa.append(log_diff.cpu().numpy())
        norm_log_diff = pr_sentence_log_probs - self.norm_log_probs
        pr_ratio = torch.exp(log_diff - torch.logsumexp(log_diff, dim=0))
        
        if paired:
            log_diff_2 = sentence_log_probs_2 - pr_sentence_log_probs_2
            if self.save:
                log_diff_2_filename = os.path.join(self.model_path, 'log_diff_2.npy')
                with NpyAppendArray(log_diff_2_filename) as npaa: 
                    npaa.append(log_diff_2.cpu().numpy())
            pr_ratio_2 = torch.exp(log_diff_2 - torch.logsumexp(log_diff_2, dim=0))
            norm_log_diff_2 = pr_sentence_log_probs_2 - self.norm_log_probs
            norm_ratio_2 = torch.exp(norm_log_diff_2 - torch.logsumexp(norm_log_diff_2, dim=0))
        
        norm_ratio = torch.exp(norm_log_diff - torch.logsumexp(norm_log_diff, dim=0))
        # ipw_loss = pr_ratio * (outcome - pred_outcome) * norm_ratio / self.pr_scale

        if self.use_pr_est:
            ipw_ratio = pr_ratio
        else:
            ipw_ratio = pr_ratio * norm_ratio / self.pr_scale

        # if self.norm_ipw_term:
                # ipw_ratio /= ipw_ratio.sum()
        if self.norm_ipw_term or not self.stabilize:
            ipw_loss = ipw_ratio * (outcome - pred_outcome)
        else:
            # if self.higheroutcomebetter:
            ipw_loss = torch.min(ipw_ratio * (outcome - pred_outcome), ipw_ratio.clamp(0.8, 1.2) * (outcome - pred_outcome))
            # else:
                # ipw_loss = torch.max(ipw_ratio * (outcome - pred_outcome), ipw_ratio.clamp(0.8, 1.2) * (outcome - pred_outcome))
        
        if self.save:
            outcome_filename = os.path.join(self.model_path, 'outcome.npy')
            pred_outcome_filename = os.path.join(self.model_path, 'pred_outcome.npy')
            with NpyAppendArray(outcome_filename) as npaa: 
                npaa.append(outcome.cpu().numpy())
            with NpyAppendArray(pred_outcome_filename) as npaa:
                npaa.append(pred_outcome.cpu().numpy())


        # rlhf_loss = pr_ratio * pred_outcome * norm_ratio
        if not paired:
            if self.norm_ipw_term or not self.stabilize:
                rlhf_loss = ipw_ratio * pred_outcome
            else:
                # if self.higheroutcomebetter:
                rlhf_loss = torch.min(ipw_ratio * pred_outcome, ipw_ratio.clamp(0.8, 1.2) * outcome)
                # else:
                    # rlhf_loss = torch.max(ipw_ratio * pred_outcome, ipw_ratio.clamp(0.8, 1.2) * outcome)
        else:
            if self.norm_rlhf_term:
                rlhf_loss = pr_ratio_2 * pred_outcome_2 * norm_ratio_2  # RLHF is much more stable when using this in the ablation (b/c not dividing by pretrained probability), but is it OK to do this b/c the pretrained prob is a constant anyway?
                # rlhf_loss = pr_ratio_2 * pred_outcome_2 * norm_ratio_2 / (pr_ratio_2*norm_ratio_2).sum()
            else:
                if not self.stabilize:
                    rlhf_loss = pr_ratio_2 * pred_outcome_2
                    # rlhf_loss = pr_ratio_2 * pred_outcome_2 / pr_ratio_2.sum()
                else:
                    # if self.higheroutcomebetter:
                    rlhf_loss = torch.min(pr_ratio_2 * pred_outcome_2, pr_ratio_2.clamp(0.8, 1.2) * pred_outcome_2)
                    # else:
                        # rlhf_loss = torch.max(pr_ratio_2 * pred_outcome_2, pr_ratio_2.clamp(0.8, 1.2) * pred_outcome_2)
            if self.save:
                pred_outcome_2_filename = os.path.join(self.model_path, 'pred_outcome_2.npy')
                with NpyAppendArray(pred_outcome_2_filename) as npaa:
                    npaa.append(pred_outcome_2.cpu().numpy())
        # pdb.set_trace()

        entropy_bonus = Categorical(logits=logits).entropy().mean(axis=-1)
        loss = self.c_rlhf*rlhf_loss + self.c_ipw*ipw_loss + self.c_entropy*entropy_bonus
        if self.higheroutcomebetter:
            loss = -loss
        loss = torch.nanmean(loss[loss != torch.inf])

        return loss
    
    def sanity_check_loss(self, logits, input_ids, outcome, pred_outcome, metric):
        token_log_probs = torch.gather(logits, dim=2, index=input_ids.unsqueeze(-1))
        token_log_probs = token_log_probs.view(-1, token_log_probs.shape[1])
        sentence_log_probs = torch.sum(token_log_probs, -1)
        sentence_probs = torch.exp(sentence_log_probs - torch.logsumexp(sentence_log_probs, dim=0))

        # pdb.set_trace()

        if self.metric == 'corr':
            if self.num_labels == 2:
                outcome[outcome == -1] = 0
            concatenated_tensors = torch.stack([sentence_probs, outcome], dim=0)
            loss = torch.corrcoef(concatenated_tensors)[0, 1]
            if torch.isnan(loss):
                loss = 0
        
        elif self.metric == 'prob':
            pr_log = torch.log(torch.tensor(self.pr_scale))
            pr_log_ratio = sentence_log_probs - pr_log
            pr_ratio = torch.exp(pr_log_ratio - torch.logsumexp(pr_log_ratio, dim=0))
            loss = pr_ratio * outcome

        # pf = torch.exp(sentence_log_probs)
        
        # ipw_loss = pf * (outcome - pred_outcome) / self.pr_scale
        # rlhf_loss = pf * pred_outcome / self.pr_scale

        # loss = rlhf_loss + ipw_loss
            loss = torch.nanmean(loss[loss != torch.inf])

        return loss

    def similarity_loss(self, finetuned_embeddings, pretrained_embeddings):
        finetuned_embeddings = F.normalize(finetuned_embeddings, dim=-1, p=2)
        pretrained_embeddings = F.normalize(pretrained_embeddings, dim=-1, p=2)

        loss = 1 - (finetuned_embeddings * pretrained_embeddings).sum(dim=-1).mean()

        return loss
    
    def kl_loss(self, logits, pretrained_logits):
        loss = F.kl_div(logits, pretrained_logits, log_target=True)

        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
            
            return (loss, None, None)

    
class CustomDataCollator(DefaultDataCollator):
    def __call__(self, examples):
        input_ids = torch.tensor([example["input_ids"] for example in examples])
        attention_mask = torch.tensor([example["attention_mask"] for example in examples])
        raw_texts = [example["raw_text"] for example in examples]
        outcomes = torch.tensor([example["outcome"] for example in examples])
        pr_log_probs = torch.tensor([example["pr_sentence_log_probs"] for example in examples])
        pred_outcomes = torch.tensor([example["pred_outcome"] for example in examples])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "raw_text": raw_texts,
            "outcome": outcomes,
            "pr_sentence_log_probs": pr_log_probs,
            "pred_outcome": pred_outcomes,
        }
    
class PairedDataCollator(DefaultDataCollator):
    def __call__(self, examples):
        input_ids = torch.tensor([example["input_ids"] for example in examples])
        attention_mask = torch.tensor([example["attention_mask"] for example in examples])
        raw_texts = [example["raw_text"] for example in examples]
        outcomes = torch.tensor([example["outcome"] for example in examples])
        input_ids_2 = torch.tensor([example["input_ids_2"] for example in examples])
        attention_mask_2 = torch.tensor([example["attention_mask_2"] for example in examples])
        pr_log_probs = torch.tensor([example["pr_sentence_log_probs"] for example in examples])
        pr_log_probs_2 = torch.tensor([example["pr_sentence_log_probs_2"] for example in examples])
        pred_outcomes = torch.tensor([example["pred_outcome"] for example in examples])
        pred_outcomes_2 = torch.tensor([example["pred_outcome_2"] for example in examples])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_ids_2": input_ids_2,
            "attention_mask_2": attention_mask_2,
            "raw_text": raw_texts,
            "outcome": outcomes,
            "pr_sentence_log_probs": pr_log_probs,
            "pr_sentence_log_probs_2": pr_log_probs_2,
            "pred_outcome": pred_outcomes,
            "pred_outcome_2": pred_outcomes_2,
        }

class TestRewardTrainer(RewardTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if not self.use_reward_data_collator:
            warnings.warn(
                "The current compute_loss is implemented for RewardDataCollatorWithPadding,"
                " if you are using a custom data collator make sure you know what you are doing or"
                " implement your own compute_loss method."
            )
        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
        )[0]
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
        )[0]
        # calculate loss, optionally modulate with margin
        pdb.set_trace()
        if "margin" in inputs:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
        else:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss
