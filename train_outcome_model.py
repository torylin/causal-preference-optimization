import pdb
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
print('Imported os', flush=True)
import torch
print('Imported torch', flush=True)
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, TrainingArguments, Trainer
print('Imported transformers', flush=True)
from datasets import load_dataset, Value
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', type=str)
    parser.add_argument('--dataset', type=str, default='hk')
    parser.add_argument('--text', type=str, default='text_full')
    parser.add_argument('--outcome', type=str, default='resp')
    parser.add_argument('--num-labels', type=int, default=1)
    parser.add_argument('--model-path', type=str, default='/projects/dataset_original/llama2/Llama-2-7b-chat-hf/')
    parser.add_argument('--model-name', type=str, default='Llama-2-7b-chat')
    parser.add_argument('--tokenizer-path', type=str)
    parser.add_argument('--results-dir', type=str)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()

    return args

def tokenize_function(examples, text):
    inputs = tokenizer(examples[text], truncation=True, padding='max_length', max_length=128)
    return inputs

def freeze_model():
    model_copy = model.clone()
    for param in model_copy.parameters():
        param.requires_grad = False
    for param in model_copy.classifier.parameters():
        param.requires_grad=True
    
    return model_copy

def rmse_metric(predictions, labels):
    mse = mean_squared_error(labels, predictions)
    rmse = torch.sqrt(torch.tensor(mse))
    return rmse

args = get_args()
output_dir = 'outcome_model'
if args.freeze:
    output_dir += '_finallayeronly'

print('Loading dataset', flush=True)
dataset = load_dataset("csv", data_files=args.csv_path)
dataset = dataset['train'].train_test_split(test_size=0.2)
remove_cols = dataset['train'].column_names
remove_cols.remove(args.text)
remove_cols.remove(args.outcome)
dataset = dataset.remove_columns(remove_cols)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device, flush=True)

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
print('Loaded tokenizer', flush=True)
model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=args.num_labels)
print('Loaded model', flush=True)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.bos_token
model.config.pad_token_id = model.config.bos_token_id

if args.freeze:
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

dataset = dataset.map(tokenize_function, batched=True, fn_kwargs={'text': args.text})
dataset = dataset.remove_columns(args.text)
dataset = dataset.rename_column(args.outcome, 'label')
if args.num_labels == 1:
    dataset = dataset.cast_column('label', Value('float32'))
elif args.num_labels == 2:
    dataset = dataset.cast_column('label', Value('int32'))

print('Tokenized data')

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, 
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1,
    bias='none')

model = get_peft_model(model, peft_config)
print('Got PEFT model')
model.print_trainable_parameters()
model.to(device)

training_args = TrainingArguments(
    output_dir=os.path.join(args.results_dir, "models/{}/{}/{}/".format(args.dataset, args.model_name, output_dir)),
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    max_grad_norm=1.0,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    logging_steps=100
)

if args.freeze:
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        model_init=freeze_model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        # compute_metrics=lambda p: rmse_metric(p.predictions, p.label_ids)
    )
else:
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        # compute_metrics=lambda p: rmse_metric(p.predictions, p.label_ids)
    )

print('Starting training')
trainer.train()


lora_path = os.path.join(args.results_dir, 'models/{}/{}/{}/lora/'.format(args.dataset, args.model_name, output_dir))
trainer.save_model(lora_path)
print('Saved LORA')

model_to_merge = PeftModel.from_pretrained(
    AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=args.num_labels).to(device), 
    lora_path)

print('Loading original model')
merged_model = model_to_merge.merge_and_unload()
print('Merged weights')
merged_model.save_pretrained(os.path.join(args.results_dir, 'models/{}/{}/{}/best_model/'.format(args.dataset, args.model_name, output_dir)), save_config=True)
print('Saved merged model')

# trained_model_path = os.path.join(args.results_dir, 'models/{}/{}/{}/best_model/'.format(args.dataset, args.model_name, output_dir))
# trainer.save_model(trained_model_path)
# model.save_pretrained(trained_model_path, save_config=True)