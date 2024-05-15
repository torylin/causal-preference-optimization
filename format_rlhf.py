import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', type=str)
    parser.add_argument('--text', type=str, default='text_full')
    parser.add_argument('--outcome', type=str, default='resp')
    parser.add_argument('--dataset', type=str, default='hk')
    parser.add_argument('--output-csv', type=str, default='HKRepData_paired.csv')
    parser.add_argument('--higheroutcomebetter', action='store_true')
    parser.add_argument('--results-dir', type=str)
    args = parser.parse_args()

    return args

def sample_pair(df, textname, outcome, higheroutcomebetter, text_dict):
    label = [0, 0]
    while label[0] == label[1]:
        pair = df.sample(n=2)
        text = pair[textname].values
        label = pair[outcome].values
    if higheroutcomebetter:
        choose_idx = np.argmax(label)
    else:
        choose_idx = np.argmin(label)
    text_dict['chosen'].append(text[choose_idx])
    text_dict['rejected'].append(text[np.abs(1-choose_idx)])
    text_dict['score_chosen'].append(label[choose_idx])
    text_dict['score_rejected'].append(label[1-choose_idx])

    return text_dict

args = get_args()

np.random.seed(231002)

df = pd.read_csv(args.csv_path)

text_dict = {'chosen': [], 'rejected': [], 'score_chosen': [], 'score_rejected': []}
for i in tqdm(range(df.shape[0])):
    text_dict = sample_pair(df, args.text, args.outcome, args.higheroutcomebetter, text_dict)

pair_df = pd.DataFrame(text_dict)

pair_df.to_csv(os.path.join(args.results_dir, '{}/train/{}'.format(args.dataset, args.output_csv)), index=False)