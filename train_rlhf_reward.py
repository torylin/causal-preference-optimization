import pdb
import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from peft.tuners.lora import _get_submodules
import peft
from trl import RewardTrainer, RewardConfig
from tqdm import tqdm
from trainer import TestRewardTrainer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', type=str)
    parser.add_argument('--dataset', type=str, default='hk')
    parser.add_argument('--model-path', type=str, default='/projects/dataset_original/llama2/Llama-2-7b-chat-hf/')
    parser.add_argument('--model-name', type=str, default='Llama-2-7b-chat')
    parser.add_argument('--results-dir', type=str)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--tokenizer-path', type=str)
    # parser.add_argument('--num-labels', type=int)
    args = parser.parse_args()

    return args

def tokenize_function(examples):
    # Tokenize the text
    chosen_inputs = tokenizer(examples['chosen'], truncation=True, padding='max_length', max_length=128)
    examples['input_ids_chosen'] = chosen_inputs['input_ids']
    examples['attention_mask_chosen'] = chosen_inputs['attention_mask']
    
    rejected_inputs = tokenizer(examples['rejected'], truncation=True, padding='max_length', max_length=128)
    examples['input_ids_rejected'] = rejected_inputs['input_ids']
    examples['attention_mask_rejected'] = rejected_inputs['attention_mask']
    
    examples['margin'] = torch.abs(torch.tensor(examples['score_chosen']) - torch.tensor(examples['score_rejected'])).view(-1, 1)

    return examples

args = get_args()
output_dir = 'rlhf_reward_model'

dataset = load_dataset("csv", data_files=args.csv_path)
dataset = dataset['train'].train_test_split(test_size=0.2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
print('Loaded tokenizer')
model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
print('Loaded model')
tokenizer.pad_token = tokenizer.bos_token
model.config.pad_token_id = model.config.bos_token_id

dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.remove_columns(['chosen', 'rejected', 'score_chosen', 'score_rejected'])
print('Tokenized data')

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, 
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1,
    bias='none',
    modules_to_save=['scores'])

training_args = RewardConfig(
    output_dir=os.path.join(args.results_dir, "models/{}/{}/{}/".format(args.dataset, args.model_name, output_dir)),
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    max_grad_norm=1.0,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    remove_unused_columns=False,
    logging_steps=100,
    max_length=128,
    gradient_checkpointing=False
)

trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    peft_config=peft_config
)

print('Starting training')
trainer.train()

lora_path = os.path.join(args.results_dir, 'models/{}/{}/{}/lora/'.format(args.dataset, args.model_name, output_dir))
trainer.save_model(lora_path)
print('Saved LORA')

model_to_merge = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(args.model_path).to(device), 
    lora_path)
print('Loading original model')
merged_model = model_to_merge.merge_and_unload()
print('Merged weights')
trained_model_path = os.path.join(args.results_dir, 'models/{}/{}/{}/best_model/'.format(args.dataset, args.model_name, output_dir))
merged_model.save_pretrained(trained_model_path, save_config=True)
print('Saved merged model')