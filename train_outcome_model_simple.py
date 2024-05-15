import pdb
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error
from joblib import dump

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', type=str)
    parser.add_argument('--dataset', type=str, default='hk')
    parser.add_argument('--text', type=str, default='text_full')
    parser.add_argument('--outcome', type=str, default='resp')
    parser.add_argument('--results-dir', type=str)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--true-outcome', type=str)
    args = parser.parse_args()

    return args

def freeze_model():
    model_copy = model.clone()
    for param in model_copy.parameters():
        param.requires_grad = False
    for param in model_copy.classifier.parameters():
        param.requires_grad=True
    
    return model_copy


args = get_args()
output_dir = 'outcome_model'
if args.freeze:
    output_dir += '_finallayeronly'

df = pd.read_csv(args.csv_path)

X = df[['numtexts', 'treatycommit', 'brave', 'evil', 'flag', 'threat', 'economy', 'treatyviolation']]
y = df[args.outcome]

if args.true_outcome is not None:
    y_true = df[args.true_outcome]
    X_train, X_test, y_train, y_test, _, y_true_test = train_test_split(X, y, y_true, test_size=0.33, random_state=42)

else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

regressor = LinearRegression()
# regressor = ElasticNet()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# Evaluate the regression model
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

if args.true_outcome is not None:
    mse_true = mean_squared_error(y_true_test, y_pred)
    print(f'MSE (true outcome): {mse_true}')

# pdb.set_trace()

if not os.path.exists(os.path.join(args.results_dir, 'models/{}/sklearn/{}/best_model/'.format(args.dataset, output_dir))):
    os.makedirs(os.path.join(args.results_dir, 'models/{}/sklearn/{}/best_model/'.format(args.dataset, output_dir)))
                
dump(regressor, os.path.join(args.results_dir, 'models/{}/sklearn/{}/best_model/linearregression.joblib'.format(args.dataset, output_dir)))