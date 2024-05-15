import pdb
import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from tqdm import tqdm
import pandas as pd
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hk')
    parser.add_argument('--model-path', type=str, default='/projects/dataset_original/llama2/Llama-2-7b-chat-hf/')
    parser.add_argument('--tokenizer-path', type=str, default='/projects/dataset_original/llama2/Llama-2-7b-chat-hf/')
    parser.add_argument('--prompt-csv', type=str)
    parser.add_argument('--prompt-col', type=str, default='text1')
    parser.add_argument('--model-name', type=str, default='Llama-2-7b-chat')
    parser.add_argument('--results-dir')
    parser.add_argument('--higheroutcomebetter', action='store_true')
    parser.add_argument('--invert', action='store_true')
    parser.add_argument('--num-labels', type=int)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    return args

def truncate_last_token_batch(batch_text):
    # Tokenize the batch of text inputs into tokens
    batch_tokens = [tokenizer.tokenize(tokenizer.eos_token+text) for text in batch_text]
    
    # Truncate the last token from each text input in the batch
    # truncated_batch_tokens = [tokens[:-5] for tokens in batch_tokens]

    # Convert the truncated tokens back to text
    # truncated_batch_text = [tokenizer.convert_tokens_to_string(tokens) for tokens in truncated_batch_tokens]
    batch_text = [tokenizer.convert_tokens_to_string(tokens) for tokens in batch_tokens]

    return batch_text

def tokenize_function(examples):
    # Tokenize the text
    trunc_text = truncate_last_token_batch(examples['prompt'])
    inputs = tokenizer(trunc_text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    # inputs = tokenizer(examples, padding='max_length', max_length=128, return_tensors='pt')
    # examples['input_ids'] = tokenizer.encode(truncate_last_token(examples['prompt']), truncation=True, padding='max_length', max_length=128, return_tensors='pt')[: input_size()]
    return inputs

args = get_args()
output_dir = 'rlhf_ppo_model'
if args.invert:
    output_dir += '_inverted'
df = pd.read_csv(args.prompt_csv)
full_prompts_list = df[args.prompt_col].values.astype(str).tolist()
# if args.dataset == 'hk':
    # prompts_list = ['The Hong Kong protesters face a serious risk of being beaten, imprisoned, or tortured by the Chinese government, but they protest anyway because',
    #                 'The 2019-2020 Hong Kong protests occurred because',
    #                 "The Hong Kong protesters seek to undermine China's sovereignity and stability, and their violence against police and destruction of property must be",
    #                 "The protesters claim that recent Chinese actions violate"]


# elif args.dataset == 'hatespeech':
    # prompts_list = ["Either we're dealing with the most incompetent elections officials or",
    #                 "These types of cunts don’t actually want",
    #                 "So why are you so jealous of white people that you have to be so racist? Obviously I understand your jealousy simply because",
    #                 " Dont run lil bitch, dont talk shit if u just gonna"]
    
# elif 'emobank' in args.dataset:
#     prompts_list = ["You are exactly the kind of person who",
#                     "Today, we serve children and families with a multitude of",
#                     "Now I understand why",
#                     "I have become more open-minded, more responsible, and, most importantly,"]
    
# full_prompts_list = prompts_list*int(500/len(prompts_list))

dataset = Dataset.from_dict({'prompt': full_prompts_list})

# dataset = load_dataset("csv", data_files=args.prompt_csv)
# remove_cols = dataset['train'].column_names
# remove_cols.remove(args.prompt_name)
# dataset = dataset.remove_columns(remove_cols)
# dataset = dataset.rename_column(args.prompt_name, 'prompt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

config = PPOConfig(
    model_name=args.model_path,
    ppo_epochs=args.epochs,
    batch_size=args.batch_size,
    remove_unused_columns=False,
    is_peft_model=True,
)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, padding_side='left')
print('Loaded tokenizer')
model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_path, peft_config=peft_config)
print('Loaded model')
tokenizer.pad_token = tokenizer.bos_token
model.config.pad_token_id = model.config.bos_token_id

reward_model = pipeline(
    'text-classification', 
    model=os.path.join(args.results_dir, 'models/{}/{}/rlhf_reward_model/best_model/'.format(args.dataset, args.model_name)), 
    tokenizer=args.tokenizer_path, 
    device=0)

input_size = LengthSampler(10, 20)

dataset = dataset.map(tokenize_function, batched=True)
# dataset['train'].set_format('pt', columns=['input_ids', 'attention_mask'], output_all_columns=True)
dataset.set_format('pt', columns=['input_ids', 'attention_mask'], output_all_columns=True)
print('Tokenized data')

# model = get_peft_model(model, peft_config)
# model.to(device)
# print('Got PEFT model')
# model.print_trainable_parameters()

trainer = PPOTrainer(
    model=model,
    config=config,
    # dataset=dataset['train'],
    dataset=dataset,
    tokenizer=tokenizer,
    # peft_config=peft_config
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.bos_token,
}

print('Starting training')
# for epoch, batch in tqdm(enumerate(trainer.dataloader), total=int(dataset['train'].num_rows/args.batch_size)):
for epoch, batch in tqdm(enumerate(trainer.dataloader), total=int(dataset.num_rows/args.batch_size)):
    try:
        query_tensors = batch["input_ids"]
    except:
        pdb.set_trace()
    
    response_tensors = trainer.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], 
                                              max_new_tokens=100, top_k=0.0, top_p=1.0, do_sample=True, min_length=5)
    # response_tensors = [trainer.generate(query_tensors[i], max_new_tokens=100, 
    #                                      top_k=0.0, top_p=1.0, do_sample=True, min_length=5
    #                                      ) for i in range(query_tensors.size(0))]
    # response_tensors = [r.squeeze() for r in response_tensors]
    # response_tensors = trainer.generate([query_tensors[i] for i in range(query_tensors.size(0))], max_new_tokens=100, top_k=0.0, top_p=1.0, do_sample=True)
    
    # response_tensor_list = []
    # texts = []
    # for i in range(response_tensors.size(0)):
    #     response_tensor_list[i] = response_tensors[i]
    #     texts[i] = tokenizer.decode(response_tensors[i], skip_special_tokens=True)

    batch["response"] = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
    # batch['response'] = texts

    # texts = [r for r in batch["response"]]
    # pipe_outputs = reward_model(texts)
    pipe_outputs = reward_model(batch['response'])
    
    # response_tensor_list = [None] * response_tensors.size(0)
    # rewards = [None] * response_tensors.size(0)
    # pred_labels = [None] * response_tensors.size(0)
    # for i in range(response_tensors.size(0)):
    #     response_tensor_list[i] = response_tensors[i]
    #     if args.num_labels == 2:
    #         pred_labels[i] = -1 if pipe_outputs[i]['label'] == 'LABEL_0' else 1
    #         pred_labels[i] = int(pred_labels[i])
    #         if args.higheroutcomebetter:
    #             rewards[i] = torch.tensor(pred_labels[i]*pipe_outputs[i]['score'])
    #         else:
    #             rewards[i] = torch.tensor(-pred_labels[i]*pipe_outputs[i]['score'])
    #     else:
    #         if args.higheroutcomebetter:
    #             rewards[i] = torch.tensor(pipe_outputs[i]['score'])
    #         else:
    #             rewards[i] = torch.tensor(-pipe_outputs[i]['score'])

    if args.num_labels == 2:
        pred_labels = np.array([output['label'] for output in pipe_outputs])
        pred_labels[pred_labels == 'LABEL_0'] = -1
        pred_labels[pred_labels == 'LABEL_1'] = 1
        pred_labels = pred_labels.astype(int)
        if args.invert:
            rewards = [torch.tensor(-pred_labels[i]*pipe_outputs[i]['score']) for i in range(len(pipe_outputs))]
        else:
            rewards = [torch.tensor(pred_labels[i]*pipe_outputs[i]['score']) for i in range(len(pipe_outputs))]
    else:
        if args.invert:
            rewards = [torch.tensor(-output["score"]) for output in pipe_outputs]
        else:
            rewards = [torch.tensor(output["score"]) for output in pipe_outputs]

    # stats = trainer.step([query_tensors[i] for i in range(query_tensors.size(0))], response_tensors, rewards)
    # pdb.set_trace()

    stats = trainer.step([query_tensors[i] for i in range(query_tensors.size(0))], 
                         [response_tensors[i] for i in range(response_tensors.size(0))], rewards)
    trainer.log_stats(stats, batch, rewards)

lora_path = os.path.join(args.results_dir, 'models/hk/{}/{}/lora/'.format(args.model_name, output_dir))
if not os.path.isdir(lora_path):
    os.makedirs(lora_path)
model.save_pretrained(lora_path)
print('Saved LORA')

model_to_merge = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(args.model_path).to(device), lora_path)
print('Loading original model')
merged_model = model_to_merge.merge_and_unload()
print('Merged weights')
trained_model_path = os.path.join(args.results_dir, 'models/{}/{}/{}/best_model/'.format(args.dataset, args.model_name, output_dir))
if not os.path.isdir(trained_model_path):
    os.makedirs(trained_model_path)
merged_model.save_pretrained(trained_model_path, save_config=True)
print('Saved model')