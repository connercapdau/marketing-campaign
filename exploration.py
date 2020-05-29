import pandas as pd
import numpy as np
import os
from marketing_code import UsedFunctions

# load data
dir = os.getcwd()
train_name = 'marketing_training.csv'
test_name = 'marketing_testing.csv'

df_train = UsedFunctions.read_data(dir, train_name)
df_test = UsedFunctions.read_data(dir, test_name)

# view data
df_train.head(5)
df_test.head(5)

# df_test first column is index so it can be dropped
df_test = df_test.drop(df_test.columns[0], axis=1)

# change responded values to 0 / 1 for no / yes since it's binary
df_train['responded_binary'] = df_train['responded'].eq('yes').mul(1)

# see how many missing values there are per column
df_train.isna().sum()
# and as a percent
df_train.isna().sum() / len(df_train) * 100

# custAge and schooling are missing lots of values
# look at mean custAge based on profession
df_train.groupby('profession', as_index=False)['custAge'].mean()
# avg age for students and retired people are significantly lower and higher, respectively, than the avg age
# replace missing age for students with 26, retired with 63, and all else with 40 (avg of all)
for i in range(len(df_train)):
    if pd.isnull(df_train.loc[i, 'custAge']) == True:
        if df_train['profession'][i] == 'student':
            df_train['custAge'][i] = 26
        elif df_train['profession'][i] == 'retired':
            df_train['custAge'][i] = 63
        else:
            df_train['custAge'][i] = 40

# schooling has an option of unknown, so all NaN values will be changed to unknown
for i in range(len(df_train)):
    if pd.isnull(df_train.loc[i, 'schooling']) == True:
        df_train['schooling'][i] = 'unknown'

# see how day of week represents responded
df_train.groupby('day_of_week', as_index=False)['responded_binary'].mean()
# they're within a few percentage points of each other so the column is dropped
df_train = df_train.drop(['day_of_week'], axis=1)


# view data types of columns
df_train.dtypes
# custAge: float, profession: object, marital: object, schooling: object, default: object, housing: object,
# loan: object, contact: object, month: object, day_of_week: object, campaign: int, pdays: int, previous: int,
# poutcome: object, emp.var.rate: float, cons.price.idx: float, cons.conf.idx: float, euribor3m: float,
# nr.employed: float, pmonths: float, pastEmail: int, responded: object

# all columns of type object are converted to categorical
for col in ['profession', 'marital', 'schooling', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']:
    df_train[col] = df_train[col].astype('category')
    df_test[col] = df_test[col].astype('category')
# responded won't be used and can be ignored since responded_binary exists
# df_train['responded'] = df_train['responded'].astype('category')

# view statistics of continuous variables
df_train_stats = df_train.describe()


# findings:
# custAge is missing approx 1800 instances. Ages are mostly between 30 to 50 years. Nothing seems wrong other than missing data
# campaign max value is much higher and right skewed
# pdays and pmonths has many people that haven't been contacted (value: 999). replace 999 values with None
# previous contact happens fewer than 25% of the time with a max of 6 times. Values seem reasonable
# emp.var.rate not missing data. Seems to be skewed to the left slightly
# cons.price.idx seems about normal
# cons.conf.idx seems to be normal
# euribor3m seems to be left skewed
# nr.employed seems normal
# pastEmail max value much higher and right skewed

# replace 999 with None in pdays and pmonths. it can be run
df_train['pdays'] = df_train['pdays'].replace(999, np.nan)
df_train['pmonths'] = df_train['pmonths'].replace(999, np.nan)

df_train_stats = df_train.describe()

# check boxplots
for col in df_train_stats.columns:
    df_train.boxplot(column=col, by='responded')

# pmonths and pdays represent the same data, so one can be dropped
# kept pmonths because 999 is a larger difference compared to people that have been contacted previously
df_train = df_train.drop(columns='pdays')
df_train_stats = df_train.describe()

# boxplots show:
# custAge: doesn't show much except older people are more likely to respond yes
# campaign: shows people contacted 5 or more times are most likely going to respond no
# pdays: doesn't have much differentiation in response answers
# previous: people that have previously responded are much more likely to respond
# emp.var.rate: lower rate has no difference for responding, but higher rates are less likely to respond
# cons.price.idx: the difference isn't substantial but lower rate more likely to respond yes
# cons.conf.idx: same as cons.price.idx
# euribor3m: lots of overlap but higher more likely to not respond; lower more likely to respond
# nr.employed: lower values are more likely to respond
# pastEmail: most people that responded had 0 to 2 emails. most people that don't respond have 0 emails