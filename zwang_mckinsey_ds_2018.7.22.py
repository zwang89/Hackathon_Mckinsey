import os
import pandas as pd
from tpot import TPOTClassifier
import numpy as np

# load data into python
syspath = os.getcwd()
train_data = syspath + "/data/train.csv"
test_data = syspath + "/data/test.csv"

# read the data
train_df = pd.read_csv(train_data)
test_df = pd.read_csv(test_data)

# check if missing data and fill missing values based features:
# 1. fill 0s for numbers of late payments
# 2. mean for application underwriting scores
pd.isnull(train_df).any()
train_df[['Count_3-6_months_late', 'Count_6-12_months_late', 'Count_more_than_12_months_late']] = \
    train_df[['Count_3-6_months_late', 'Count_6-12_months_late', 'Count_more_than_12_months_late']].fillna(value=0)

train_df[['application_underwriting_score']] = train_df[['application_underwriting_score']].\
    fillna(train_df[['application_underwriting_score']].mean())


pd.isnull(test_df).any()
test_df[['Count_3-6_months_late', 'Count_6-12_months_late', 'Count_more_than_12_months_late']] = \
    test_df[['Count_3-6_months_late', 'Count_6-12_months_late', 'Count_more_than_12_months_late']].fillna(value=0)

test_df[['application_underwriting_score']] = test_df[['application_underwriting_score']].\
    fillna(test_df[['application_underwriting_score']].mean())

# check any missing values
pd.isnull(train_df).any()
pd.isnull(test_df).any()


# data engineering for training dataset
# new feature created

# 1. create new features: age in years
age_in_years = round(train_df['age_in_days'] / 365, 2)
train_df['age_in_years'] = age_in_years


# 2. create new feature: total late payment
count_total_late = train_df['Count_3-6_months_late'] + train_df['Count_6-12_months_late'] \
                   + train_df['Count_more_than_12_months_late']
train_df['count_total_late'] = count_total_late


# 3. create new feature: any late payment
late_payments_any = []

def late_payment():
    """
    create a new list that indicate if a client has previous late payment or not
    """
    for i in range(0, len(train_df)):
        if max(train_df['Count_3-6_months_late'][i], train_df['Count_6-12_months_late'][i],
               train_df['Count_more_than_12_months_late'][i]) > 0:
            late = 1
        else:
            late = 0
        late_payments_any.append(late)
late_payment()

train_df['late_payments_any'] = late_payments_any

# 4. create new feature: prior renewal
prior_renewal = []

def prior_ren():
    """if total number of late payments and # of paid prinums greater than 12 months"""
    for i in range(0, len(train_df)):
        if (train_df['Count_3-6_months_late'][i]+ train_df['Count_6-12_months_late'][i] +
            train_df['Count_more_than_12_months_late'][i] + train_df['no_of_premiums_paid'][i]) > 12:
            prior_renewal_temp = 1
        else:
            prior_renewal_temp = 0
        prior_renewal.append(prior_renewal_temp)


prior_ren()

train_df['prior_renewal'] = prior_renewal


# 5. create new feature: prior renewal without any late payment
prior_renewal_no_late = []

def prior_ren_no_late():
    """# of paid premiums greater than 12 months
    and no any late payment in 3-6 months and 6-12 months"""
    for i in range(0, len(train_df)):
        if ((max(train_df['Count_3-6_months_late'][i], train_df['Count_6-12_months_late'][i]) == 0 and
             train_df['no_of_premiums_paid'][i]) > 12):
            prior_renewal_no_late_temp = 1
        else:
            prior_renewal_no_late_temp = 0
        prior_renewal_no_late.append(prior_renewal_no_late_temp)

prior_ren_no_late()
train_df['prior_renewal_no_late'] = prior_renewal_no_late


# 6. create new feature: late payment in all window periods
late_payment_in_all = []

def late_payment_all():
    """have late payments for all period windows: 3-6, 6-12, 12 +"""

    for i in range(0, len(train_df)):
        if (train_df['Count_3-6_months_late'][i] > 0 and
            train_df['Count_6-12_months_late'][i]>0 and
            train_df['Count_more_than_12_months_late'][i]>0):
            late_all = 1
        else:
            late_all = 0
        late_payment_in_all.append(late_all)

late_payment_all()
train_df['late_payment_in_all'] = late_payment_in_all


# 7. create new feature: % of late payment

pct_late_payment = train_df['count_total_late'] / \
                   (train_df['count_total_late']+train_df['no_of_premiums_paid'])
train_df['pct_late_payment'] = pct_late_payment

# 8. create new feature: total revenue from a customer
income_premium_pcnt = []
def income_premium():

    for i in range(0, len(train_df)):
        pnct = train_df['premium'][i] / train_df['Income'][i]
        income_premium_pcnt.append(pnct)

income_premium()

train_df['income_premium_pcnt'] = income_premium_pcnt

# 9 create new feature: total revenue from a customer
total_revenue = []
def total_re():

    for i in range(0, len(train_df)):
        revenue = train_df['premium'][i] * train_df['no_of_premiums_paid'][i]
        total_revenue.append(revenue)

total_re()

train_df['total_revenue'] = total_revenue


# data pre-processing for test dataset

# 1. create dummy variables for sourcing channel
train_df = pd.get_dummies(train_df, columns=['sourcing_channel'], drop_first=True)

# 2. get dummy variable for residence location
train_df = pd.get_dummies(train_df, columns=['residence_area_type'], drop_first=True)


###########################################################
# data engineering for test dataset
# new feature created

# 1. create new features: age in years
age_in_years = round(test_df['age_in_days'] / 365, 2)
test_df['age_in_years'] = age_in_years

# 2. create new feature: total late payment
count_total_late = test_df['Count_3-6_months_late'] + test_df['Count_6-12_months_late'] \
                   + test_df['Count_more_than_12_months_late']
test_df['count_total_late'] = count_total_late

# 3. create new feature: any late payment
late_payments_any = []


def late_payment():
    """
    create a new list that indicate if a client has previous late payment or not
    """
    for i in range(0, len(test_df)):
        if max(test_df['Count_3-6_months_late'][i], test_df['Count_6-12_months_late'][i],
               test_df['Count_more_than_12_months_late'][i]) > 0:
            late = 1
        else:
            late = 0
        late_payments_any.append(late)


late_payment()

test_df['late_payments_any'] = late_payments_any

# 4. create new feature: prior renewal
prior_renewal = []


def prior_ren():
    """if total number of late payments and # of paid prinums greater than 12 months"""
    for i in range(0, len(test_df)):
        if (test_df['Count_3-6_months_late'][i] + test_df['Count_6-12_months_late'][i] +
            test_df['Count_more_than_12_months_late'][i] + test_df['no_of_premiums_paid'][i]) > 12:
            prior_renewal_temp = 1
        else:
            prior_renewal_temp = 0
        prior_renewal.append(prior_renewal_temp)


prior_ren()

test_df['prior_renewal'] = prior_renewal

# 5. create new feature: prior renewal without any late payment
prior_renewal_no_late = []


def prior_ren_no_late():
    """# of paid premiums greater than 12 months
    and no any late payment in 3-6 months and 6-12 months"""
    for i in range(0, len(test_df)):
        if ((max(test_df['Count_3-6_months_late'][i], test_df['Count_6-12_months_late'][i]) == 0 and
             test_df['no_of_premiums_paid'][i]) > 12):
            prior_renewal_no_late_temp = 1
        else:
            prior_renewal_no_late_temp = 0
        prior_renewal_no_late.append(prior_renewal_no_late_temp)


prior_ren_no_late()
test_df['prior_renewal_no_late'] = prior_renewal_no_late

# 6. create new feature: late payment in all window periods
late_payment_in_all = []


def late_payment_all():
    """have late payments for all period windows: 3-6, 6-12, 12 +"""

    for i in range(0, len(test_df)):
        if (test_df['Count_3-6_months_late'][i] > 0 and
                test_df['Count_6-12_months_late'][i] > 0 and
                test_df['Count_more_than_12_months_late'][i] > 0):
            late_all = 1
        else:
            late_all = 0
        late_payment_in_all.append(late_all)


late_payment_all()
test_df['late_payment_in_all'] = late_payment_in_all

# 7. create new feature: % of late payment

pct_late_payment = test_df['count_total_late'] / \
                   (test_df['count_total_late'] + test_df['no_of_premiums_paid'])
test_df['pct_late_payment'] = pct_late_payment

# 8. create new feature: total revenue from a customer
income_premium_pcnt = []


def income_premium():
    for i in range(0, len(test_df)):
        pnct = test_df['premium'][i] / test_df['Income'][i]
        income_premium_pcnt.append(pnct)


income_premium()

test_df['income_premium_pcnt'] = income_premium_pcnt

# 9 create new feature: total revenue from a customer
total_revenue = []


def total_re():
    for i in range(0, len(test_df)):
        revenue = test_df['premium'][i] * test_df['no_of_premiums_paid'][i]
        total_revenue.append(revenue)


total_re()

test_df['total_revenue'] = total_revenue

# data pre-processing for train dataset

# 1. create dummy variables for sourcing channel
test_df = pd.get_dummies(test_df, columns=['sourcing_channel'], drop_first=True)

# 2. get dummy variable for residence location
test_df = pd.get_dummies(test_df, columns=['residence_area_type'], drop_first=True)

########################################################################################

# train and fit the model
# training validation split: split training and validation
X_train = train_df.drop(columns='renewal')
y_train = train_df['renewal']

tpot = TPOTClassifier(generations=2, scoring='roc_auc', population_size=100, verbosity=2, cv=5)
tpot.fit(X_train, y_train)
tpot.export('tpot_cl_pipeline_v1.py')

##########################################################################################


# get the best pipeline from tpot, parameters, change to different models
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(learning_rate=0.1, max_depth=5,
                                 max_features=0.7500000000000001,
                                 min_samples_leaf=19, min_samples_split=8,
                                 n_estimators=100, subsample=1.0)


clf.fit(X_train, y_train)
a = clf.predict(test_df)
b = clf.predict_proba(test_df)
test_df['prediction'] = pd.Series(a, index=test_df.index)
b_prob = pd.DataFrame(b, columns=['prob_0', 'prob_1'])
test_df.reset_index(drop=True, inplace=True)
b_prob.reset_index(drop=True, inplace=True)
test_df_predict = pd.concat([test_df, b_prob], axis=1)

# define the functions for hours-incentives, incentives-proba, and final revnenue
# define function for hours and incencts
def hrs_incent(incentive):
    hours = 10 * (1-np.exp(-incentive/400))
    return hours


# define function for improving prob and hours
def improve_prob(hours):
    prob = 20 *(1-np.exp(-hours/5))/100
    return prob

# define function for net revenue
def net_revenue(i, incentives):
    return (test_df_predict['prob_1'][i] + improve_prob(hrs_incent(incentives)))*test_df_predict['premium'][i] - incentives


# define function to cal incentives
incentives = []
net_rev = []

def find_incentives():
    """for each of policy, find their best net revenue and the incentives"""
    for i in range(0, len(test_df_predict)):
        rev = []  # create new temp variable to store all possible revenue
        for incentive in range(0, 2000):
            tem_rev = net_revenue(i, incentive)  # cal all possible revenue
            rev.append(tem_rev)  # store all revenue

        opt_incent = rev.index(max(rev)) + 1  # find the best incentives based on max(rev)
        incentives.append(opt_incent)  # get the list of incentives for policy
        print('processing the', i, 'th data')


find_incentives()
print(len(incentives))

# change the col to correct names
policy_prob_incent =  test_df_predict.filter(['id', 'prob_1']).rename(index=str, columns={"prob_1": "renewal"})
policy_prob_incent['incentives'] = incentives

# export files
policy_prob_incent.to_csv('policy_prob_incent_final.csv', index=False)