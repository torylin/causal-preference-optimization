import pdb
import os
import torch
# print('Imported torch', flush=True)
import argparse
from trainer import CustomTrainer, CustomDataCollator, PairedDataCollator
# print('Imported trainer', flush=True)
# from dataset import CustomDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, Trainer, EvalPrediction
# print('Imported transformers', flush=True)
# from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset, Value
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from tqdm import tqdm
import pandas as pd
import csv
import wandb

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', type=str)
    parser.add_argument('--dataset', type=str, default='hk')
    parser.add_argument('--scaling', type=float, default=0.003)
    parser.add_argument('--text', type=str, default='text_full')
    parser.add_argument('--outcome', type=str, default='resp')
    parser.add_argument('--model-path', type=str, default='/projects/dataset_original/llama2/Llama-2-7b-chat-hf/')
    parser.add_argument('--model-name', type=str, default='Llama-2-7b-chat')
    parser.add_argument('--tokenizer-path', type=str, default='/projects/dataset_original/llama2/Llama-2-7b-chat-hf/')
    parser.add_argument('--opt-type', type=str, default='bias_corrected_rlhf')
    parser.add_argument('--reg-type', type=str, default='none')
    parser.add_argument('--outcome-model-path', type=str)
    parser.add_argument('--num-labels', type=int)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--results-dir', type=str)
    parser.add_argument('--invert-loss', action='store_true')
    parser.add_argument('--clm', action='store_true')
    parser.add_argument('--clm-highoutcome', action='store_true')
    parser.add_argument('--clm-lowoutcome', action='store_true')
    parser.add_argument('--higheroutcomebetter', action='store_true')
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--paired', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--metric', type=str, default='corr')
    parser.add_argument('--c-ipw', type=float, default=1.0)
    parser.add_argument('--c-rlhf', type=float, default=1.0)
    parser.add_argument('--c-entropy', type=float, default=0)
    parser.add_argument('--output-name', type=str)
    parser.add_argument('--norm-rlhf-term', action='store_true')
    parser.add_argument('--stabilize', action='store_true')
    parser.add_argument('--norm-ipw-term', action='store_true')
    parser.add_argument('--ipw01', action='store_true')
    parser.add_argument('--noclm', action='store_true')
    parser.add_argument('--use-pr-est', action='store_true')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--eval-dir', type=str)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.1)
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()

    return args

def tokenize_function(examples, text, outcome, clm, paired, second_text):
    # Tokenize the text
    inputs = tokenizer([str(ex) for ex in examples[text]], truncation=True, padding='max_length', max_length=128)
    # Add the 'outcome' field to the inputs
    if not clm:
        inputs['raw_text'] = examples[text]
        inputs['outcome'] = examples[outcome]
        inputs['pr_sentence_log_probs'] = examples['pr_sentence_log_probs']
        inputs['pred_outcome'] = examples['pred_outcome']
    if paired:
        inputs2 = tokenizer([str(ex) for ex in examples[second_text]], truncation=True, padding='max_length', max_length=128)
        inputs['input_ids_2'] = inputs2['input_ids']
        inputs['attention_mask_2'] = inputs2['attention_mask']
        inputs['pr_sentence_log_probs_2'] = examples['pr_sentence_log_probs_2']
        inputs['pred_outcome_2'] = examples['pred_outcome_2']
    return inputs

def gradient_clipping_callback(optimizers, model, inputs, **kwargs):
    max_grad_norm = 1.0  # Set the maximum gradient norm as needed
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)


def sanity_check(p: EvalPrediction):
    predictions = p.predictions
    pdb.set_trace()

args = get_args()

if args.clm:
    output_dir = 'clm'
    clm = TrainingArguments
elif args.clm_highoutcome:
    output_dir = 'clm_highoutcome'
    clm = True
elif args.clm_lowoutcome:
    output_dir = 'clm_lowoutcome'
    clm = True
else:
    output_dir = 'bias_corrected_rlhf'
    if args.norm_rlhf_term:
        output_dir += '_normed'
    if args.stabilize:
        output_dir += '_stabilized'
        # output_dir += '_stabilizedrlhfterm'
    if args.norm_ipw_term:
        output_dir += '_ipwnormed'
    if args.c_entropy > 0:
        output_dir += '_entropybonus{}'.format(args.c_entropy)
    if args.c_ipw == 0.0:
        output_dir += '_ablationrlhf'
        if args.paired:
            output_dir += 'paired'
            
    clm = False
    if args.load_in_8bit:
        output_dir += '_8bit'
    if args.ipw01:
        output_dir += '_ipw01'
    if args.noclm:
        output_dir += '_noclm'

if args.invert_loss:
    print('Reversing direction of optimization')
    output_dir += '_inverted'
#     if args.reg_type == 'kl':
#         output_dir = 'inverted_ce_kl_loss_scaling{}'.format(args.scaling)
#     elif args.reg_type == 'sim':
#         output_dir = 'inverted_ce_sim_loss_scaling{}'.format(args.scaling)
# else:
#     if args.reg_type == 'kl':
#         output_dir = 'ce_kl_loss_scaling{}'.format(args.scaling)
#     elif args.reg_type == 'sim':
#         output_dir = 'ce_sim_loss_scaling{}'.format(args.scaling)

print(output_dir)

dataset = load_dataset("csv", data_files=args.csv_path)

if clm:
    remove_cols = dataset['train'].column_names
    remove_cols.remove(args.text)
    dataset = dataset.remove_columns(remove_cols)
else:
    if args.num_labels == 1:
        dataset = dataset.cast_column(args.outcome, Value('float32'))
    elif args.num_labels == 2:
        dataset = dataset.cast_column(args.outcome, Value('int32'))

if args.opt_type != 'sanity_check':
    dataset = dataset['train'].train_test_split(test_size=0.2, seed=240402)
    # pdb.set_trace()
pr_scale = dataset['train'].num_rows

# gc.collect()
# torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# model_path = "facebook/opt-125m"
# model_path = "facebook/opt-2.7b"
# model_path = '/projects/dataset_original/llama2/Llama-2-7b-chat-hf/'
# model_path = 'facebook/opt-6.7b'
# model_name = 'Llama-2-7b-chat'
# model_name = 'opt-125m'
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
print('Loaded tokenizer')
model = AutoModelForCausalLM.from_pretrained(args.model_path)
print('Loaded model')
tokenizer.pad_token = tokenizer.bos_token
model.config.pad_token_id = model.config.bos_token_id

dataset = dataset.map(tokenize_function, batched=True, fn_kwargs={'text': args.text, 'outcome': args.outcome, 'clm': clm, 
                                                                  'paired': args.paired, 'second_text': '{}2_trunc_completion'.format(args.text)})
if clm:
    dataset = dataset.remove_columns([args.text])
print('Tokenized data')

# dataset = CustomDataset(csv_path, tokenizer, max_length=128)

# for param in model.parameters():
#     param.requires_grad = False
#     if param.ndim == 1:
#         param.data = param.data.to(torch.float32)
# model.gradient_checkpointing_enable()
# model.enable_input_require_grads()
# class CastOutputToFloat(nn.Sequential):
#   def forward(self, x): return super().forward(x).to(torch.float32)
# model.lm_head = CastOutputToFloat(model.lm_head)

# config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules=["q_proj", "v_proj"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM"
# )

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=args.lora_rank, 
    lora_alpha=args.lora_alpha, 
    lora_dropout=args.lora_dropout)

model = get_peft_model(model, peft_config)
print('Got PEFT model')
model.print_trainable_parameters()

pdb.set_trace()

training_args = TrainingArguments(
    output_dir=os.path.join(args.results_dir, "models/{}/{}/{}/".format(args.dataset, args.model_name, output_dir)),
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    # per_device_train_batch_size=1,
    max_grad_norm=1.0,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    remove_unused_columns=False,
    logging_steps=100,
    learning_rate=args.lr
)


# Create the data collator for language modeling
if clm:
    print('Training CLM model')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test']
    )
else:
    if args.paired:
        data_collator = PairedDataCollator()
    else:
        data_collator = CustomDataCollator()
    # if args.eval:
    #     eval_dataset = dataset['train']
    # else:
    #     train_dataset = dataset['train']
    #     eval_dataset = dataset['test']
    #     trainer = CustomTrainer(
    #         model=model,
    #         pretrained_model_path=args.model_path,
    #         scaling=args.scaling,
    #         invert=args.invert_loss,
    #         reg_type=args.reg_type,
    #         opt_type=args.opt_type,
    #         pr_scale=pr_scale,
    #         model_name=args.model_name,
    #         dataset=args.dataset,
    #         higheroutcomebetter=args.higheroutcomebetter,
    #         outcome_model_path=args.outcome_model_path,
    #         num_labels=args.num_labels,
    #         results_dir=args.results_dir,
    #         eightbit=args.load_in_8bit,
    #         paired=args.paired,
    #         metric=args.metric,
    #         c_ipw=args.c_ipw,
    #         c_rlhf=args.c_rlhf,
    #         c_entropy=args.c_entropy,
    #         norm_rlhf_term=args.norm_rlhf_term,
    #         stabilize=args.stabilize,
    #         norm_ipw_term=args.norm_ipw_term,
    #         use_pr_est=args.use_pr_est,
    #         model_path=args.model_path,
    #         save=args.save,
    #         args=training_args,
    #         data_collator=data_collator,
    #         eval_dataset=dataset['train']
    #     )
    # else:
    if args.wandb:
        trainer = CustomTrainer(
            model=model,
            pretrained_model_path=args.model_path,
            scaling=args.scaling,
            invert=args.invert_loss,
            reg_type=args.reg_type,
            opt_type=args.opt_type,
            pr_scale=pr_scale,
            model_name=args.model_name,
            dataset=args.dataset,
            higheroutcomebetter=args.higheroutcomebetter,
            outcome_model_path=args.outcome_model_path,
            num_labels=args.num_labels,
            results_dir=args.results_dir,
            eightbit=args.load_in_8bit,
            paired=args.paired,
            metric=args.metric,
            c_ipw=args.c_ipw,
            c_rlhf=args.c_rlhf,
            c_entropy=args.c_entropy,
            norm_rlhf_term=args.norm_rlhf_term,
            stabilize=args.stabilize,
            norm_ipw_term=args.norm_ipw_term,
            use_pr_est=args.use_pr_est,
            model_path=args.model_path,
            save=args.save,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            report_to='wandb'
        )
    else:
        trainer = CustomTrainer(
            model=model,
            pretrained_model_path=args.model_path,
            scaling=args.scaling,
            invert=args.invert_loss,
            reg_type=args.reg_type,
            opt_type=args.opt_type,
            pr_scale=pr_scale,
            model_name=args.model_name,
            dataset=args.dataset,
            higheroutcomebetter=args.higheroutcomebetter,
            outcome_model_path=args.outcome_model_path,
            num_labels=args.num_labels,
            results_dir=args.results_dir,
            eightbit=args.load_in_8bit,
            paired=args.paired,
            metric=args.metric,
            c_ipw=args.c_ipw,
            c_rlhf=args.c_rlhf,
            c_entropy=args.c_entropy,
            norm_rlhf_term=args.norm_rlhf_term,
            stabilize=args.stabilize,
            norm_ipw_term=args.norm_ipw_term,
            use_pr_est=args.use_pr_est,
            model_path=args.model_path,
            save=args.save,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test']
        )
# trainer.add_callback(gradient_clipping_callback)

if args.eval:
    results = trainer.evaluate(eval_dataset=dataset['test'])
    print('Sanity check {}: {}'.format(args.metric, results['eval_loss']))
    row = [args.output_name, results['eval_loss']]

    output_csv = '{}_{}_{}.csv'.format(args.dataset, args.metric, args.model_name)
    if not os.path.exists(os.path.join(args.eval_dir, output_csv)):
        with open(os.path.join(args.eval_dir, output_csv), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['model', args.metric])
            writer.writerow(row)
    else:
        with open(os.path.join(args.eval_dir, output_csv), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    mean_outcome = pd.read_csv(args.csv_path)[args.outcome].values.mean()
    print('Mean outcome: {}'.format(mean_outcome))
else:
    # Start training
    print('Starting training')
    trainer.train()
    # optimizer = AdamW(model.parameters(), lr=1e-3)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)

    # max_grad_norm = 5.0
    # for epoch in range(training_args.num_train_epochs):
    #     for step, batch in tqdm(enumerate(trainer.get_train_dataloader())):
    #         # Forward pass
    #         loss = trainer.compute_loss(model, batch)

    #         # Backward pass with gradient clipping
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Apply gradient clipping

    #         # Update model parameters
    #         optimizer.step()
    #         optimizer.zero_grad()

    lora_path = os.path.join(args.results_dir, 'models/{}/{}/{}/lora/'.format(args.dataset, args.model_name, output_dir))
    trainer.save_model(lora_path)
    print('Saved LORA')

    model_to_merge = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(args.model_path).to(device), lora_path)
    print('Loading original model')
    merged_model = model_to_merge.merge_and_unload()
    print('Merged weights')
    trained_model_path = os.path.join(args.results_dir, 'models/{}/{}/{}/best_model/'.format(args.dataset, args.model_name, output_dir))
    merged_model.save_pretrained(trained_model_path, save_config=True)
    print('Saved merged model to {}'.format(trained_model_path))

