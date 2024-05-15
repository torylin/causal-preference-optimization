import torch
import os
import argparse
import pdb
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, pipeline, AutoModelForSequenceClassification
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import csv
import sys
from empath import Empath
import nltk
from nltk.corpus import stopwords
from joblib import load

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--outcome-model-path', type=str)
    parser.add_argument('--tokenizer-path', type=str)
    parser.add_argument('--prompt-csv', type=str)
    parser.add_argument('--prompt-col', type=str, default='text')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-length', type=int, default=200)
    parser.add_argument('--num-prompts', type=int)
    parser.add_argument('--results-dir', type=str)
    parser.add_argument('--output-name', type=str, default='clm')
    parser.add_argument('--dataset', type=str, default='hatespeech')
    parser.add_argument('--generate-text', action='store_true')
    parser.add_argument('--generate-outcomes', action='store_true')
    parser.add_argument('--completion-csv', type=str)
    parser.add_argument('--completion-col', type=str, default='completion')
    parser.add_argument('--num-labels', type=int, default=2)
    parser.add_argument('--output-csv', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--combine-prompt-df', action='store_true')
    parser.add_argument('--seed', type=int, default=240122)
    parser.add_argument('--do-not-append', action='store_true')
    parser.add_argument('--data-dir', type=str)

    args = parser.parse_args()

    return args

def encode_empath(df, text_col, lexicon):
    rows = []
    for i in tqdm(range(df.shape[0])):
        try:
            cat_counts = lexicon.analyze(df[text_col][i], categories=hk_categories)
            sentences = nltk.sent_tokenize(df[text_col][i])
            cat_counts['numtexts'] = len(sentences)
        except:
            cat_counts = lexicon.analyze('', categories=hk_categories)
            cat_counts['numtexts'] = 0
        rows.append(cat_counts)

    df_empath = pd.DataFrame(rows)
    df_empath[df_empath > 1] = 1
    df_empath.columns = ['treatycommit', 'brave', 'evil', 'flag', 'threat', 'economy', 'treatyviolation', 'numtexts']

    return df_empath

args = get_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
df_res_exists = False
if args.output_dir is None:
    output_dir = args.results_dir
else:
    output_dir = args.output_dir
    
if args.output_csv is not None:
    if os.path.exists(args.output_csv):
        df_res = pd.read_csv(args.output_csv)
        df_res_exists = True
else:
    output_csv = os.path.join(output_dir, '{}_{}_{}_generated_outputs.csv'.format(args.dataset, args.output_name, args.model_name))
    if os.path.exists(output_csv):
        df_res = pd.read_csv(output_csv)
        df_res_exists = True

if args.generate_text:
    prompt_df = pd.read_csv(args.prompt_csv)
    prompt_df = prompt_df.dropna(subset=args.prompt_col)
    print('Found prompt df with {} rows'.format(prompt_df.shape[0]))
    if df_res_exists:
        print('Found {} existing rows of generated outputs'.format(df_res.shape[0]))
        df_prompt_merged = pd.merge(prompt_df, df_res, how='outer', left_on=args.prompt_col, right_on='prompt', indicator=True)
        df_prompt_merged = df_prompt_merged[df_prompt_merged['_merge'] != 'both']
        df_prompt_new = df_prompt_merged[prompt_df.columns]
        n_all = df_res.shape[0] + df_prompt_new.shape[0]
        print('{} prompts remaining of {}'.format(df_prompt_new.shape[0], n_all))
        if (args.num_prompts == None) or (args.num_prompts == n_all):
            prompt_df = df_prompt_new
        elif args.num_prompts > n_all:
            n_extra = args.num_prompts - n_all
            prompt_df_extra = prompt_df.sample(n=n_extra, replace=True, random_state=args.seed)
            prompt_df = pd.concat([df_prompt_new, prompt_df_extra], ignore_index=True)
        elif df_res.shape[0] < args.num_prompts < n_all:
            prompt_df = df_prompt_new[:args.num_prompts - df_res.shape[0]]
        elif args.num_prompts <= df_res.shape[0]:
            sys.exit()
        print('Generating outputs for {} more prompts'.format(prompt_df.shape[0]))

    else:
        if args.num_prompts is not None:
            # pdb.set_trace()
            if args.num_prompts < prompt_df.shape[0]:
                prompt_df = prompt_df[:args.num_prompts]
                print('Generating {} new rows for {} total'.format(args.num_prompts - prompt_df.shape[0], args.num_prompts))
            elif args.num_prompts > prompt_df.shape[0]:
                n_extra = args.num_prompts - prompt_df.shape[0]
                print('Generating {} new rows for {} total'.format(n_extra, args.num_prompts))
                prompt_df_extra = prompt_df.sample(n=n_extra, replace=True, random_state=args.seed)
                prompt_df = pd.concat([prompt_df, prompt_df_extra], ignore_index=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, padding_side='left')
    print('Loaded tokenizer')
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    print('Loaded model')
    model.to(device)
    print('Model to GPU')
    tokenizer.pad_token = tokenizer.bos_token
    model.config.pad_token_id = model.config.bos_token_id

    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    prompts_list = prompt_df[args.prompt_col].dropna().values.tolist()
    # prompts = Dataset.from_dict({'prompt': prompts_list})
    prompt_batches = [prompts_list[i:i+args.batch_size] for i in range(0, len(prompts_list), args.batch_size)]
    generated_completions = []

    results_dict = {'prompt': prompts_list, 'generated_text': [], 'completion': []}


    for batch in tqdm(prompt_batches):
        batch_texts = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        generated_batch = model.generate(**batch_texts, max_length=args.max_length, 
                                        #  top_k=0.0, top_p=1.0, 
                                         do_sample=True, min_length=5)
        generated_completions.extend(generated_batch)

    for i, completion in tqdm(enumerate(generated_completions), total=len(prompts_list)):
        decoded_output = tokenizer.decode(completion, skip_special_tokens=True)
        text_diff = decoded_output.split(results_dict['prompt'][i])[-1]
        results_dict['generated_text'].append(decoded_output)
        results_dict['completion'].append(text_diff)

    # for generated_batch in tqdm(text_generator(KeyDataset(prompts, 'prompt'), max_length=args.max_length)):
    #     results_dict['completion'].extend(generated_batch[0]['generated_text'])

    results_df = pd.DataFrame(results_dict)
    del(tokenizer)
    del(model)

if args.generate_outcomes:

    if args.completion_csv != None:
        results_df = pd.read_csv(args.completion_csv)

    if ('sklearn' in args.outcome_model_path) and ('hk' in args.dataset):
        outcome_model = load(args.outcome_model_path)
        lexicon = Empath()
        hk_categories = ['commitment', 'bravery', 'mistreatment', 'flags', 'threat', 'economy', 'violation']
        csvs = ['HKarmstreatyobligation.csv', 'HKarmsbrave.csv', 'HKarmsevil.csv', 'HKarmsflag.csv', 'HKarmsthreat.csv', 'HKarmseconomy.csv', 'HKarmstreatyviolation.csv']
        stop_words = set(stopwords.words('english'))
        for i in range(len(hk_categories)):
            hk_csv = pd.read_csv(os.path.join(args.data_dir, 'hk/hk_rct/{}'.format(csvs[i])), header=0, names=['text'])
            words = ' '.join(hk_csv.text.values.tolist()).split()
            words = [word for word in words if word.lower() not in stop_words]
            # words = list(set(words))
            lexicon.create_category(hk_categories[i], [hk_categories[i]] + words, model='nytimes')
        
        df_empath = encode_empath(results_df, args.completion_col, lexicon)
        df_empath = df_empath[['numtexts', 'treatycommit', 'brave', 'evil', 'flag', 'threat', 'economy', 'treatyviolation']]

        generated_outcomes = outcome_model.predict(df_empath)
    
    else:
    
        outcome_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        print('Loaded tokenizer')
        outcome_model = AutoModelForSequenceClassification.from_pretrained(args.outcome_model_path, num_labels=args.num_labels)
        print('Loaded model')
        outcome_model.to(device)
        print('Model to GPU')
        outcome_tokenizer.pad_token = outcome_tokenizer.bos_token
        outcome_model.config.pad_token_id = outcome_model.config.bos_token_id

        completions_list = results_df[args.completion_col].dropna().values.tolist()
        # completions = Dataset.from_dict({'completion': completions_list})
        completion_batches = [completions_list[i:i+args.batch_size] for i in range(0, len(completions_list), args.batch_size)]
        generated_outcomes = []

        outcome_dict = {'generated_outcome': []}

        for batch in tqdm(completion_batches):
            batch_texts = outcome_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            outcome_output = outcome_model(**batch_texts, output_hidden_states=True)
            outcome_logits = outcome_output.logits.detach()
            if args.num_labels == 1:
                pred_outcome = outcome_logits.squeeze()
            elif args.num_labels == 2:
                pred_outcome = torch.argmax(outcome_logits, dim=-1).type(torch.float)
                idx = pred_outcome.clone().type(torch.int64)
                pred_outcome[pred_outcome == 0] = -1
                pred_probs = torch.softmax(outcome_logits, dim=-1)
                pred_probs = pred_probs.gather(dim=1, index=idx.view(-1, 1)).squeeze()
                pred_outcome *= pred_probs

            generated_outcomes.extend(pred_outcome.cpu().tolist())

    results_df['generated_outcomes'] = pd.Series(generated_outcomes)

    # mean_output_csv = '{}_{}_mean_outcome.csv'.format(args.dataset, args.model_name)
    # if not os.path.exists(os.path.join(output_dir, mean_output_csv)):
    #     with open(os.path.join(output_dir, mean_output_csv), 'w') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(['model', 'mean_outcome'])
    #         writer.writerow(row)
    # else:
    #     with open(os.path.join(output_dir, mean_output_csv), 'a') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(row)

if args.combine_prompt_df:
    prompt_df['{}_generated_text'.format(args.prompt_col)] = results_df['generated_text']
    prompt_df['{}_completion'.format(args.prompt_col)] = results_df['completion']
    results_df = prompt_df

results_df.dropna(inplace=True)

if (df_res_exists) and (not args.do_not_append):
    results_df = pd.concat([df_res, results_df], ignore_index=True)

row = [args.output_name, results_df['generated_outcomes'].mean()]
mean_output_csv = '{}_{}_mean_outcome.csv'.format(args.dataset, args.model_name)
if not os.path.exists(os.path.join(output_dir, mean_output_csv)):
    with open(os.path.join(output_dir, mean_output_csv), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'mean_outcome'])
        writer.writerow(row)
else:
    with open(os.path.join(output_dir, mean_output_csv), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)

if args.output_csv == None:
    results_df.to_csv(output_csv, index=False)
else:
    results_df.to_csv(args.output_csv, index=False)