import pandas as pd
import numpy as np
import pdb
from scipy import stats
import pingouin as pg
from sklearn.metrics import cohen_kappa_score
# from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
# import krippendorff as kd
from irrCAC.raw import CAC
# from rpy2.robjects import DataFrame, FloatVector, IntVector
# from rpy2.robjects.packages import importr
import warnings
warnings.filterwarnings("ignore")

# r_icc = importr("agreement")

def binary_corr(x, y):
    n_11 = np.sum((x == 1) & (y == 1))
    n_00 = np.sum((x == 0) & (y == 0))
    n_10 = np.sum((x == 1) & (y == 0))
    n_01 = np.sum((x == 0) & (y == 1))

    correlation = (n_11 * n_00 - n_10 * n_01) / np.sqrt((n_11 + n_10) * (n_11 + n_01) * (n_00 + n_10) * (n_00 + n_01))
    return correlation

# HUMAN WIN RATE

np.random.seed(240119)
survey_path = ''

df = pd.read_csv(survey_path)
df = df.iloc[2:,]
df_filtered = df.filter(regex='HK_pretrainedvsours').dropna(how='all')
df_rater_cols = df_filtered.transpose()
df_filtered = pd.melt(df_filtered, value_vars=df_filtered.columns, var_name='question', value_name='response')

theirs_pref_count = np.sum(df_filtered['response']=='A')
ours_pref_count = np.sum(df_filtered['response']=='B')
theirs_win_rate = theirs_pref_count / (theirs_pref_count + ours_pref_count)

df_answered = df_filtered[(df_filtered['response'] == 'A') | (df_filtered['response'] == 'B')]
df_answered.loc[df_answered['response'] == 'A', 'response'] = 0 # theirs
df_answered.loc[df_answered['response'] == 'B', 'response'] = 1 # ours

se = df_answered.response.sem()

print('Human theirs win rate: {:.3f} [{:.3f}, {:.3f}]'.format(theirs_win_rate, theirs_win_rate-1.96*se, theirs_win_rate+1.96*se))
print('se: {:.3f}'.format(se))

df_majority = df_answered.groupby('question')['response'].apply(lambda x: x.mode().sample(n=1).iloc[0]).reset_index()
majority_theirs_win_rate = 1-df_majority.response.mean()
majority_se = df_majority.response.sem()
print('Majority human theirs win rate: {:.3f} [{:.3f}, {:.3f}]'.format(
    majority_theirs_win_rate, 
    majority_theirs_win_rate-1.96*majority_se, 
    majority_theirs_win_rate+1.96*majority_se))
print('se: {:.3f}'.format(majority_se))

# GPT WIN RATE & RELIABILITY

gpt_path = ''
df_gpt = pd.read_csv(gpt_path)
df_gpt['question'] = ['{}_HK_pretrainedvsours'.format(i) for i in range(1, df_gpt.shape[0]+1)]
df_merged = pd.merge(df_majority, df_gpt, on='question', how='inner')
theirs_gpt_pref = df_merged.pretrained_idx == df_merged.pref
ours_gpt_pref = ~theirs_gpt_pref.values
gpt_se = stats.sem(theirs_gpt_pref)
theirs_gpt_win_rate = np.mean(theirs_gpt_pref)
print('GPT theirs win rate (on these samples): {:.3f} [{:.3f}, {:.3f}]'.format(
    theirs_gpt_win_rate,
    theirs_gpt_win_rate-1.96*gpt_se, 
    theirs_gpt_win_rate+1.96*gpt_se))
print('se: {:.3f}'.format(gpt_se))

ours_human_pref = df_merged.response.values

maj_human_gpt_corr = binary_corr(ours_human_pref, ours_gpt_pref)
print('Majority human-GPT corr: {:.3f}'.format(maj_human_gpt_corr))

# maj_human_gpt_kappa = cohen_kappa_score(ours_human_pref, ours_gpt_pref)
# print("Majority human-GPT Cohen's kappa: {:.3f}".format(maj_human_gpt_kappa))

df_merged['response_gpt'] = ours_gpt_pref
df_merged_subset = df_merged[['question', 'response', 'response_gpt']]
df_merged_subset.set_index('question', inplace=True)
df_cac = CAC(df_merged_subset)
maj_human_gpt_fleiss = df_cac.fleiss()
print("Majority human-GPT Fleiss' kappa: {:.3f} [{:.3f}, {:.3f}]".format(
    maj_human_gpt_fleiss['est']['coefficient_value'],
    maj_human_gpt_fleiss['est']['confidence_interval'][0],
    maj_human_gpt_fleiss['est']['confidence_interval'][1]))
# df_merged_subset = df_merged[['question', 'response', 'response_gpt']]
# df_merged_subset = pd.melt(df_merged_subset, id_vars=['question'], value_vars=['response', 'response_gpt'], var_name='rater', value_name='rating')
# df_merged_subset.loc[df_merged_subset['rater'] == 'response', 'rater'] = 'human_majority'
# df_merged_subset.loc[df_merged_subset['rater'] == 'response_gpt', 'rater'] = 'gpt'

# maj_human_icc = pg.intraclass_corr(data=df_merged_subset, targets='question', raters='rater', ratings='rating')

# INTER-RATER RELIABILITY

respondent_dfs = {}

for id in df.PROLIFIC_PID:
    df_respondent = df[df.PROLIFIC_PID == id]
    df_respondent_filtered = df_respondent.filter(regex='HK_pretrainedvsours')
    df_respondent_filtered = pd.melt(df_respondent_filtered, value_vars=df_respondent_filtered.columns, var_name='question', value_name='response')
    df_respondent_answered = df_respondent_filtered[(df_respondent_filtered['response'] == 'A') | (df_respondent_filtered['response'] == 'B')]
    df_respondent_answered.loc[df_respondent_answered['response'] == 'A', 'response'] = 0 # theirs
    df_respondent_answered.loc[df_respondent_answered['response'] == 'B', 'response'] = 1 # ours
    respondent_dfs[id] = df_respondent_answered

full_respondent_df = pd.concat([df.assign(id=id_val) for id_val, df in respondent_dfs.items()], ignore_index=True)

corrs = {'corr': [], 'kappa': [], 'n': []}
for id1, df1 in respondent_dfs.items():
    for id2, df2 in respondent_dfs.items():
        if id1 < id2:
            df_corr = pd.merge(df1, df2, on='question', how='inner')
            # if df_corr.shape[0] > 1:
            corrs['corr'].append(binary_corr(df_corr.response_x, df_corr.response_y))
            corrs['kappa'].append(cohen_kappa_score(df_corr.response_x.values.astype(bool), df_corr.response_y.values.astype(bool)))
            corrs['n'].append(df_corr.shape[0])

corrs['corr'] = np.array(corrs['corr'])
corrs['kappa'] = np.array(corrs['kappa'])
corrs['n'] = np.array(corrs['n'])

n_corr = corrs['n'][~np.isnan(corrs['corr'])]
n_kappa = corrs['n'][~np.isnan(corrs['kappa'])]
corrs['corr'] = corrs['corr'][~np.isnan(corrs['corr'])]
corrs['kappa'] = corrs['kappa'][~np.isnan(corrs['kappa'])]

human_human_corr = np.sum(corrs['corr']*n_corr) / np.sum(n_corr)
print('Human-human corr: {:.3f}'.format(human_human_corr))

# human_human_kappa = np.sum(corrs['kappa']*n_kappa) / np.sum(n_kappa)
# print("Human-human Cohen's kappa: {:.3f}".format(human_human_kappa))

df_cac = CAC(df_rater_cols)
human_human_fleiss = df_cac.fleiss()
print("Human-human Fleiss' kappa: {:.3f} [{:.3f}, {:.3f}]".format(
    human_human_fleiss['est']['coefficient_value'],
    human_human_fleiss['est']['confidence_interval'][0],
    human_human_fleiss['est']['confidence_interval'][1]))

# GPT-RATER RELIABILITY

gpt_corrs = {'corr': [], 'kappa': [], 'lower_kappa_ci': [], 'upper_kappa_ci': [], 'n': []}
for id1, df1 in respondent_dfs.items():
    df_corr = pd.merge(df1, df_gpt, on='question', how='outer')
    theirs_gpt_pref = df_corr.pretrained_idx == df_corr.pref
    ours_gpt_pref = ~theirs_gpt_pref.values
    df_corr['response_gpt'] = ours_gpt_pref
    df_corr = df_corr[['question', 'response', 'response_gpt']]
    df_corr.set_index('question', inplace=True)
    if df_corr.shape[0] > 0:
        df_cac = CAC(df_corr)
        gpt_fleiss = df_cac.fleiss()
        gpt_corrs['corr'].append(binary_corr(df_corr.response, ours_gpt_pref))
        # gpt_corrs['kappa'].append(cohen_kappa_score(df_corr.response.values.astype(bool), ours_gpt_pref.astype(bool)))
        gpt_corrs['kappa'].append(gpt_fleiss['est']['coefficient_value'])
        gpt_corrs['lower_kappa_ci'].append(gpt_fleiss['est']['confidence_interval'][0])
        gpt_corrs['upper_kappa_ci'].append(gpt_fleiss['est']['confidence_interval'][1])
        gpt_corrs['n'].append(df_corr.shape[0])

gpt_corrs['corr'] = np.array(gpt_corrs['corr'])
gpt_corrs['n'] = np.array(gpt_corrs['n'])
gpt_corrs['kappa'] = np.array(gpt_corrs['kappa'])
n_corr = gpt_corrs['n'][~np.isnan(gpt_corrs['corr'])]
n_kappa = gpt_corrs['n'][~np.isnan(gpt_corrs['kappa'])]
gpt_corrs['corr'] = gpt_corrs['corr'][~np.isnan(gpt_corrs['corr'])]
gpt_corrs['kappa'] = gpt_corrs['kappa'][~np.isnan(gpt_corrs['kappa'])]

gpt_human_corr = np.sum(gpt_corrs['corr']*n_corr) / np.sum(n_corr)
print('GPT-human corr: {:.3f}'.format(gpt_human_corr))

gpt_human_kappa = np.sum(gpt_corrs['corr']*n_kappa) / np.sum(n_kappa)
print('GPT-human kappa: {:.3f}'.format(gpt_human_kappa))


pdb.set_trace()