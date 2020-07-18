from google.colab import drive
drive.mount('/content/drive/')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
!pip install sweetviz==1.0a7
import sweetviz
train=pd.read_csv("/content/drive/My Drive/data/server/Train.csv")
test=pd.read_csv("/content/drive/My Drive/data/server/Test.csv")
#from google.colab import output
#output.clear()
#Performing sweetviz which will show all eda process
##Analyzing a single dataframe (and its optional target feature)
my_report1 = sweetviz.analyze([train, "Train"],target_feat='MULTIPLE_OFFENSE')
my_report1.show_html("Report.html") 
#Comparing two dataframes (e.g. Test vs Training sets)
my_report2 = sweetviz.compare([train, "Train"], [test, "Test"], "MULTIPLE_OFFENSE")
my_report2.show_html("Report.html")
train.head()
test.head()
#feature preprocessing
train.drop(train.columns[[0,1]], axis = 1, inplace = True)
test.drop(test.columns[[0,1]], axis = 1, inplace = True)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print(scaler.fit(train))
print(scaler.fit(train))
train.X_12.value_counts
train.X_12.isna().sum()
test.X_12.value_counts
test.X_12.isna().sum()
#imputing missing values
from sklearn.impute import KNNImputer
knn_imputer = KNNImputer(n_neighbors=2, weights="uniform")
train['X_12'] = knn_imputer.fit_transform(train[['X_12']])
test['X_12'] = knn_imputer.fit_transform(test[['X_12']])
train.shape
test.shape
test.info()
(train.MULTIPLE_OFFENSE).value_counts()
train.drop(train.columns[[0,2,3,4,5,6,7,8,12,13]],axis=1,inplace=True)
test.drop(test.columns[[0,2,3,4,5,6,7,8,12,13]],axis=1,inplace=True)
train.info()
test.info()
#Got good accuracy in xgboost model so building it
import xgboost as xgb1


X = train.iloc[:,0:4].values
Y = train['MULTIPLE_OFFENSE'] 
#building xgboost model to check the important feature without assigning any parameter
#xgb1 = xgb.XGBClassifier()
#xgb1.fit(X,Y)
#columns_ = train.iloc[:, 0:15].columns
#(pd.Series(xgb1.feature_importances_,index=columns_ ).nlargest(5).plot(kind='barh'))

#checking data is balanced or not
(train.MULTIPLE_OFFENSE).value_counts()
#data is imbalanced so performing smote
from imblearn.combine import SMOTETomek
oversample = SMOTETomek(random_state=42)
X_train_res, y_train_res = oversample.fit_sample(X, Y)
#checking the shape of data after performing smote
print(X_train_res.shape)

print(y_train_res.shape)

print(sum(y_train_res==1))

print(sum(y_train_res==0))
# building model
#performing hyper parameter tuning 
xgb=xgb1.XGBClassifier()
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

from sklearn.model_selection import  StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
folds = 5
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X,Y), verbose=3, random_state=1001 )


random_search.fit(X, Y)
print(random_search.cv_results_)
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)
#the best paramater value we will take and put it inmodel

xgb=xgb1.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1.0, gamma=1,
              learning_rate=0.02, max_delta_step=0, max_depth=4,
              min_child_weight=1, missing=None, n_estimators=600, n_jobs=1,
              nthread=1, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=True, subsample=0.8, verbosity=1)


xgb.fit(X_train_res, y_train_res)
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score,recall_score,accuracy_score, f1_score, precision_score
train_pred=xgb.predict(X)

#train splitted data checking the accuracy

matrix1=confusion_matrix(Y,train_pred)
print(matrix1)
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(Y, train_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))



Predction1=xgb.predict(test)

Final_pred = pd.DataFrame(Final_pred)


Final_pred.info()
test1=pd.read_csv("/content/drive/My Drive/data/server/Test.csv")
Final_pred.head()
submission = pd.concat([ test1, Final_pred], axis=1)
submission.head()
mapping = {submission.columns[17]:'MULTIPLE_OFFENSE'}

submission=submission.rename(columns=mapping)
submission = submission.iloc[:,0:18]
submission.head()
submission['MULTIPLE_OFFENSE'].value_counts



print(submission)

submission.to_csv('/content/drive/My Drive/data/server/submission.csv',index=False)