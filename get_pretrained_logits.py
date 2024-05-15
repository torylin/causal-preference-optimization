import os
import torch
import torch.nn.functional as F
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import csv
import pdb
import gc
from joblib import load
import nltk
from nltk.corpus import stopwords
from empath import Empath
import numpy as np
from scipy.special import expit

# nltk.download('punkt')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', type=str)
    parser.add_argument('--output-csv', type=str)
    parser.add_argument('--dataset', type=str, default='hk')
    parser.add_argument('--text', type=str, default='text_full')
    parser.add_argument('--model-path', type=str, default='/projects/dataset_original/llama2/Llama-2-7b-chat-hf/')
    parser.add_argument('--model-name', type=str, default='Llama-2-7b-chat')
    parser.add_argument('--tokenizer-path', type=str, default='/projects/dataset_original/llama2/Llama-2-7b-chat-hf/')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--results-dir', type=str)
    parser.add_argument('--paired', action='store_true')
    parser.add_argument('--outcome-model-path', type=str)
    parser.add_argument('--num-labels', type=int)
    parser.add_argument('--skip-logits', action='store_true')
    parser.add_argument('--confounding', action='store_true')
    parser.add_argument('--beta', type=float)
    parser.add_argument('--intercept', type=float)
    parser.add_argument('--c-coef', type=float)
    parser.add_argument('--data-dir', type=str)
    args = parser.parse_args()

    return args

def tokenize_function(examples, text, paired, second_text):
    # Tokenize the text
    inputs = tokenizer([str(ex) for ex in examples[text]], return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    # Add the 'outcome' field to the inputs
    if paired:
        inputs2 = tokenizer([str(ex) for ex in examples[second_text]], return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        inputs['input_ids_2'] = inputs2['input_ids']
        inputs['attention_mask_2'] = inputs2['attention_mask']
    return inputs

def encode_empath(df, text_col, lexicon):
    rows = []
    for i in tqdm(range(df.shape[0])):
        cat_counts = lexicon.analyze(df[args.text][i], categories=hk_categories)
        sentences = nltk.sent_tokenize(df[args.text][i])
        cat_counts['numtexts'] = len(sentences)
        rows.append(cat_counts)

    df_empath = pd.DataFrame(rows)
    df_empath[df_empath > 1] = 1
    df_empath.columns = ['treatycommit', 'brave', 'evil', 'flag', 'threat', 'economy', 'treatyviolation', 'numtexts']

    return df_empath

args = get_args()

dataset = load_dataset("csv", data_files=args.csv_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
print('Loaded tokenizer')
tokenizer.pad_token = tokenizer.bos_token

if not args.skip_logits:
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    print('Loaded model')
    model.config.pad_token_id = model.config.bos_token_id

    model.eval()

dataset = dataset.map(tokenize_function, batched=True, 
                      fn_kwargs={'text': args.text, 
                                 'paired': args.paired, 
                                 'second_text': '{}2_trunc_completion'.format(args.text)})

input_ids = torch.tensor(dataset['train']['input_ids']).to(device)
attention_mask = torch.tensor(dataset['train']['attention_mask']).to(device)
if args.paired:
    input_ids_2 = torch.tensor(dataset['train']['input_ids_2']).to(device)
    attention_mask_2 = torch.tensor(dataset['train']['attention_mask_2']).to(device)

n = dataset['train'].shape[0]
pred_outcomes = torch.empty(n).to(device)
if args.paired:
    pred_outcomes_2 = torch.empty(n).to(device)

if not args.skip_logits:
    all_log_probs = torch.empty(n).to(device)

    if args.paired:
        all_log_probs_2 = torch.empty(n).to(device)
        
    for i in tqdm(range(0, n, args.batch_size)):

        batch_input_ids = input_ids[i:i+args.batch_size]
        batch_attention_mask = attention_mask[i:i+args.batch_size]
        if args.paired:
            batch_input_ids_2 = input_ids_2[i:i+args.batch_size]
            batch_attention_mask_2 = attention_mask_2[i:i+args.batch_size]

        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, output_hidden_states=True)
        pretrained_logits = outputs.logits.detach()
        pr_token_log_probs = torch.gather(pretrained_logits, dim=2, index=batch_input_ids.unsqueeze(-1))
        pr_token_log_probs = pr_token_log_probs.view(-1, pr_token_log_probs.shape[1])
        pr_sentence_log_probs = torch.sum(pr_token_log_probs, -1)

        all_log_probs[i:i+args.batch_size] = pr_sentence_log_probs

        if args.paired:
            outputs2 = model(input_ids=batch_input_ids_2, 
                            attention_mask=batch_attention_mask_2,
                            output_hidden_states=True)
            pretrained_logits_2 = outputs2.logits.detach()
            pr_token_log_probs_2 = torch.gather(pretrained_logits_2, dim=2, index=batch_input_ids_2.unsqueeze(-1))
            pr_token_log_probs_2 = pr_token_log_probs_2.view(-1, pr_token_log_probs_2.shape[1])
            pr_sentence_log_probs_2 = torch.sum(pr_token_log_probs_2, -1)

            all_log_probs_2[i:i+args.batch_size] = pr_sentence_log_probs_2

    del(model)
    torch.cuda.empty_cache()

if ('sklearn' in args.outcome_model_path) and ('hk' in args.dataset):
    df = pd.read_csv(args.csv_path)
    X = df[['numtexts', 'treatycommit', 'brave', 'evil', 'flag', 'threat', 'economy', 'treatyviolation']]

    model = load(args.outcome_model_path)
    pred_outcomes = model.predict(X)

    if args.paired:
        lexicon = Empath()
        hk_categories = ['commitment', 'bravery', 'mistreatment', 'flags', 'threat', 'economy', 'violation']
        csvs = ['HKarmstreatyobligation.csv', 'HKarmsbrave.csv', 'HKarmsevil.csv', 'HKarmsflag.csv', 'HKarmsthreat.csv', 'HKarmseconomy.csv', 'HKarmstreatyviolation.csv']
        stop_words = set(stopwords.words('english'))
        for i in range(len(hk_categories)):
            csv = pd.read_csv(os.path.join(args.data_dir, 'hk/hk_rct/{}'.format(csvs[i]), header=0, names=['text']))
            words = ' '.join(csv.text.values.tolist()).split()
            words = [word for word in words if word.lower() not in stop_words]
            # words = list(set(words))
            lexicon.create_category(hk_categories[i], [hk_categories[i]] + words, model='nytimes')
        
        df_empath = encode_empath(df, '{}2_trunc_completion'.format(args.text), lexicon)
        df_empath = df_empath[['numtexts', 'treatycommit', 'brave', 'evil', 'flag', 'threat', 'economy', 'treatyviolation']]

        pred_outcomes_2 = model.predict(df_empath)

    if args.confounding:
        coef_dict = {'treatycommit': 2.68,
            'brave': 1.85,
            'evil': 0.14,
            'flag': -2.12,
            'threat': -2.07,
            'economy': -0.94,
            'treatyviolation': 0.75}
        
        coefs = np.array(list(coef_dict.values()))
        df_coefs = df[coef_dict.keys()]*coef_dict.values()
        df_coefs = df_coefs.sum(axis=1)
        df_coefs *= args.c_coef
        df_coefs += args.intercept
        probs = df_coefs.apply(expit, axis=0).values

        confounders = np.zeros(len(probs))
        for i in range(len(probs)):
            confounders[i] = np.random.choice([-1, 1], size=1, p=[1-probs[i], probs[i]])

        res_var = np.var(df.resp.values - pred_outcomes)/len(probs)
        noise = np.random.normal(0, res_var, len(probs))
        confounding = args.beta*confounders + noise
        pred_outcomes = pred_outcomes + confounding

        if args.paired:
            df_coefs_2 = df_empath[coef_dict.keys()]*coef_dict.values()
            df_coefs_2 = df_coefs_2.sum(axis=1)
            df_coefs_2 *= args.c_coef
            df_coefs_2 += args.intercept
            probs_2 = df_coefs_2.apply(expit, axis=0).values

            confounders_2 = np.zeros(len(probs_2))
            for i in range(len(probs_2)):
                confounders_2[i] = np.random.choice([-1, 1], size=1, p=[1-probs_2[i], probs_2[i]])
            noise_2 = np.random.normal(0, res_var, len(probs_2))

            confounding_2 = args.beta*confounders_2 + noise_2
            pred_outcomes_2 = pred_outcomes_2 + confounding_2

            df = df.rename(columns={'pred_outcome_2': 'resp_pred_2'})

        df = df.rename(columns={'pred_outcome': 'resp_pred'})

else:

    model = AutoModelForSequenceClassification.from_pretrained(
        args.outcome_model_path, num_labels=args.num_labels).to(device)
    print('Loaded outcome model')
    model.config.pad_token_id = model.config.bos_token_id
    model.eval()

    for i in tqdm(range(0, n, args.batch_size)):
        batch_input_ids = input_ids[i:i+args.batch_size]
        batch_attention_mask = attention_mask[i:i+args.batch_size]
        if args.paired:
            batch_input_ids_2 = input_ids_2[i:i+args.batch_size]
            batch_attention_mask_2 = attention_mask_2[i:i+args.batch_size]

        outcome_output = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, output_hidden_states=True)
        outcome_logits = outcome_output.logits.detach()
        if args.num_labels == 1:
            pred_outcome = outcome_logits.squeeze()
        elif args.num_labels == 2:
            pred_probs = F.softmax(outcome_logits, dim=-1)
            idx = torch.argmax(pred_probs, dim=-1)
            pred_probs = pred_probs.gather(dim=1, index=idx.view(-1, 1)).squeeze()
            idx[idx == 0] = -1
            pred_outcome = pred_probs*idx

        pred_outcomes[i:i+args.batch_size] = pred_outcome

        if args.paired:
            outcome_output_2 = model(input_ids=batch_input_ids_2, attention_mask=batch_attention_mask_2, output_hidden_states=True)
            outcome_logits_2 = outcome_output_2.logits.detach()
            if args.num_labels == 1:
                pred_outcome_2 = outcome_logits_2.squeeze()
            elif args.num_labels == 2:
                pred_probs_2 = F.softmax(outcome_logits_2, dim=-1)
                idx_2 = torch.argmax(pred_probs_2, dim=-1)
                pred_probs_2 = pred_probs_2.gather(dim=1, index=idx_2.view(-1, 1)).squeeze()
                idx_2[idx_2 == 0] = -1
                pred_outcome_2 = pred_probs_2*idx_2

            pred_outcomes_2[i:i+args.batch_size] = pred_outcome_2

if not args.confounding:
    df = pd.read_csv(args.csv_path)

if not args.skip_logits:
    df['pr_sentence_log_probs'] = all_log_probs.cpu()
    if args.paired:
        df['pr_sentence_log_probs_2'] = all_log_probs_2.cpu()

if ('sklearn' in args.outcome_model_path) and ('hk' in args.dataset):
    df['pred_outcome'] = pred_outcomes
    if args.paired:
        df['pred_outcome_2'] = pred_outcomes_2
else:
    df['pred_outcome'] = pred_outcomes.cpu()
    if args.paired:
        df['pred_outcome_2'] = pred_outcomes_2.cpu()

print('Generated for {} rows'.format(df.shape[0]))

if args.output_csv == None:
    df.to_csv(args.csv_path, index=False)
else:
    df.to_csv(args.output_csv, index=False)