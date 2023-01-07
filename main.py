import pandas as pd
import os
import requests
import sklearn

if not os.path.exists('../Data'):
    os.mkdir('../Data')

if 'nba2k-full.csv' not in os.listdir('../Data'):
    print('Train dataset loading.')
    url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/nba2k-full.csv', 'wb').write(r.content)
    print('Loaded.')

data_path = "../Data/nba2k-full.csv"


def clean_data(dp):
    df = pd.read_csv(dp)
    df['b_day'] = pd.to_datetime(df['b_day'], format='%m/%d/%y')
    df['draft_year'] = pd.to_datetime(df['draft_year'], format='%Y')
    df['team'] = df['team'].fillna('No Team')
    df['height'] = df['height'].apply(lambda x: float(x.split('/')[1]))
    df['weight'] = df['weight'].apply(lambda x: float(x.split('/')[1].replace('kg.', '')))
    df['salary'] = df['salary'].apply(lambda x: float(x.replace('$', '')))
    df['country'] = df['country'].apply(lambda x: 'Not-USA' if x != 'USA' else 'USA')
    df['draft_round'] = df['draft_round'].apply(lambda x: '0' if x == 'Undrafted' else '1')
    return df


def feature_data(cdf):
    cdf['version'] = pd.to_datetime(cdf['version'].apply(lambda x: x.replace('NBA2k', '20')), format='%Y')
    cdf['age'] = pd.DatetimeIndex(cdf['version']).year - pd.DatetimeIndex(cdf['b_day']).year
    cdf['experience'] = pd.DatetimeIndex(cdf['version']).year - pd.DatetimeIndex(cdf['draft_year']).year
    cdf['bmi'] = cdf['weight'] / cdf['height']**2
    cdf.drop(['version', 'b_day', 'draft_year', 'weight', 'height'], axis=1, inplace=True)
    for i in cdf.columns:
        if cdf[i].nunique() > 50 and i not in ['age', 'experience', 'bmi', 'salary']:
            cdf.drop(i, axis=1, inplace=True)
    return cdf


def multicol_data(fd):
    X = fd.drop(columns='salary')
    y = fd.salary
    m = X.corr(numeric_only=True)
    pairs = []
    for i in range(m.shape[0]):
        for j in range(i + 1, m.shape[0]):
            if abs(m.iloc[i][j]) > 0.5:
                pairs.append((i, j))
    for p in pairs:
        col0 = m.columns[p[0]]
        col1 = m.columns[p[1]]
        r0 = y.corr(X[col0])
        r1 = y.corr(X[col1])
        if r0 < r1:
            fd = fd.drop(columns=col0)
        else:
            fd = fd.drop(columns=col1)
    return fd


if __name__ == '__main__':
    path = "../Data/nba2k-full.csv"
    df_cleaned = clean_data(path)
    df_featured = feature_data(df_cleaned)
    df = multicol_data(df_featured)
    print(list(df.select_dtypes('number').drop(columns='salary')))
