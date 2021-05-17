"""
>>>
    PROJECT: IFRS9 LGD MODEL
    DATE CREATED: 2021-05-14
    AUTHOR: LONGPH5
>>>
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, auc, roc_curve, roc_auc_score, precision_recall_curve
import statsmodels.api as sm
import graphviz
from joblib import dump, load
from xverse.transformer import WOE, MonotonicBinning
from xverse.graph import BarCharts
from xverse.ensemble import VotingSelector



# Load file data
dt = pd.read_csv(r'\\ho-file01\QTRR\Noibo\PHONG MHRR\LongPH5\8. LGD Model\Export_LGD_Data_v1.csv'\
                 ,dtype={'CUSTOMER_ID':'str'})
    
""" 
>>>
    Filter Workout and FTD_DATE > '2013-01-31' and FTD_DATE < '2019-01-01'
    and EAD > 500000
    and IFRS9_SEGMENT = 'Retail'
    and remove (SOURCE = CC and FTD_DATE in 2014-09-30','2019-10-31)
>>>
"""


# Filter satisfied data
date = ['2014-09-30','2014-10-31'] 
dt = dt.query("EAD >= 50000 \
              and FTD_DATE > '2013-01-31' \
              and FTD_DATE < '2019-01-01'\
              and TAG_CURE == 'Workout' \
              and IFRS9_SEGMENT == 'Retail' \
              and not (FTD_DATE in @date and SOURCE == 'CC') ")
  
    
  
# Random Sampling using stratify     
dt['full_loss'] = dt['LGD_FINAL'].apply(lambda x: 1 if x == 1 else 0)
X_name =['EAD', 'SOURCE', 'MIA','TOTAL_OS_UNSECURED',\
        'AVG_OS_UNSECURED_L3M', 'AVG_OS_UNSECURED_L6M', 'AVG_OS_UNSECURED_L12M',\
        'AVG_OS_CONTRACT_L3M', 'AVG_OS_CONTRACT_L6M', 'AVG_OS_CONTRACT_L12M',\
        'TERM', 'MOB', 'LIMIT_CC_UTILIZATION', 'CURENT_LIMIT',\
        'LIMIT_CC_UTILIZATION_L3M', 'LIMIT_CC_UTILIZATION_L6M',\
        'EAD_TO_DISBURSEMENT_AMT', 'OS_TO_DISBURSEMENT_AMT', 'NBR_USED_PRODUCT',\
        'DPD_CONTRACT_10_L6M', 'DPD_CONTRACT_30_L6M', 'DPD_CONTRACT_60_L6M',\
        'DPD_CONTRACT_10_L12M', 'DPD_CONTRACT_30_L12M', 'DPD_CONTRACT_60_L12M',\
        'DPD_10_L6M', 'DPD_30_L6M', 'DPD_60_L6M', 'DPD_10_L12M', 'DPD_30_L12M',\
        'DPD_60_L12M', 'YEAR_RELATION', 'LEFT_CENSOR_12M']

y = dt['full_loss']
X = dt[X_name]


# Weight of evidence
"""

Information Value Variable Predictiveness

Less than 0.02     Not useful for prediction
0.02 to 0.1        Weak predictive Power  
0.1 to 0.3         Medium predictive Power  
0.3 to 0.5         Strong predictive Power  
>0.5               Suspicious Predictive Power

"""
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 100, stratify = y)   
y_train.value_counts()
y_test.value_counts()


# Long-list variable
clf = VotingSelector()
clf.fit(X_train, y_train)
print(clf.available_techniques)
feature_important = clf.feature_importances_
feature_vote = clf.feature_votes_


# Short-list_variable
short_list = feature_vote.query('Votes > 3')['Variable_Name']


# Weight of evidence of short list
clf_WoE = WOE(treat_missing = 'separate')
clf_WoE.fit(X[short_list], y )


# Information value
iv_df = clf_WoE.iv_df

# Weight of evidence
woe_df = clf_WoE.woe_df

# Plot WOE chart
clf_plot = BarCharts(bar_type='v')
clf_plot.plot(woe_df)

# Transform WOE
X_WOE = clf_WoE.transform(X_train[short_list])




# Fit model
sm_model = sm.Logit(y_train, X_WOE)
fitted_model = sm_model.fit()
fitted_model.summary()


# Take variables which p-value <= 0.05
cols = ['MOB','YEAR_RELATION','TERM','AVG_OS_CONTRACT_L12M','AVG_OS_CONTRACT_L3M','CURENT_LIMIT','LEFT_CENSOR_12M','LIMIT_CC_UTILIZATION','EAD','DPD_10_L6M']
    

# Refit model
sm_model = sm.Logit(y_train, X_WOE[cols])
fitted_model = sm_model.fit()
fitted_model.summary()

# Predict on training set
y_pred = fitted_model.predict(X_train[cols])
prediction = list(map(round,y_pred))
cm = confusion_matrix(y_train, prediction)
cm
fpr, tpr, thresholds = roc_curve(y_train, y_pred)
roc_auc = roc_auc_score(y_train, y_pred)
roc_auc
gini = 2*roc_auc - 1
gini

print(classification_report(y_train, prediction))

# Precision Recall Cureve
precision, recall, thresholds = precision_recall_curve(y_train, y_pred)


plt.title('ROC (Receiver Operating Characteristic) - Training Data')
plt.plot(fpr, tpr, 'b', label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate (TPR)')
plt.xlabel('False Positive Rate (FPR)')
plt.show()


# Predict on test set
y_pred = fitted_model.predict(X_test[cols])
prediction = list(map(round,y_pred))
fpr, tpr, thresholds = roc_curve(y_test, y_pred, 0.5)
roc_auc = roc_auc_score(y_test, y_pred)
gini = 2*roc_auc - 1
gini

print(classification_report(y_test, prediction))

plt.title('ROC (Receiver Operating Characteristic) - Testing Data')
plt.plot(fpr, tpr, 'b', label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate (TPR)')
plt.xlabel('False Positive Rate (FPR)')
plt.show()





















# # Class Exploratory Data Analysis 
# class Exploratory_Data_Analysis():
    
#     def __init__(self, data):
#         self.df = data
#         self.shape = self.df.shape
#         self.columns = self.df.columns
#         self.statistic = self.df.describe().T
#         self.missing = self.df.isna().sum()
           
#     def missing_imputation(self):
#         self.df = self.df.fillna(0)
#         return self
      
#     def hist(self, x, bins):
#         hist = sns.histplot(self.df[x], bins = bins)
#         return hist



             







