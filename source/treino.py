import pandas as pd
import numpy as np
import time
import sys
import sagemaker

from sagemaker.pytorch import PyTorch


prefix = "sagemaker/dt-CTA"
sagemaker_session = sagemaker.Session()

role = sys.argv[1]
bucket = sys.argv[2]
stack_name = sys.argv[3]
commit_id = sys.argv[4]
commit_id = commit_id[0:7]

timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
job_name = stack_name + "-" + commit_id + "-" + timestamp


from random import sample
import random

from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn import metrics
#from IPython.display import Image
#from pydotplus import graph_from_dot_data

df = pd.read_csv('Telescope data.csv')

np.random.seed(42)


df['Class_num'] = np.where(df['Class'] == 'g', 1.0, 0.0)

df = df.drop(['Class'],axis=1)
col_names = df.columns.tolist()

train, test = np.split( df.sample(frac=1), [int(.8*len(df))])

print(train.shape, test.shape)

# Separate target and predictors
y_train = train['Class_num']
x_train = train.drop(['Class_num'], axis=1)
y_test = test['Class_num']
x_test = test.drop(['Class_num'], axis=1)

all_vars = x_train.columns.tolist()
top_5_vars = ['fAlpha', 'fLength', 'fWidth', 'fSize', 'fM3Long']
bottom_vars = [cols for cols in all_vars if cols not in top_5_vars]

# Drop less important variables leaving the top_5
x_train    = x_train.drop(bottom_vars, axis=1)
x_test     = x_test.drop(bottom_vars, axis=1)


tree_model = tree.DecisionTreeClassifier(max_depth=3)
# Fit a decision tree
tree_model = tree_model.fit(x_train, y_train)
# Training accuracy
tree_model.score(x_train, y_train)

# Predictions/probs on the test dataset
predicted = pd.DataFrame(tree_model.predict(x_test))
probs = pd.DataFrame(tree_model.predict_proba(x_test))

# Store metrics
tree_accuracy = metrics.accuracy_score(y_test, predicted)
tree_roc_auc = metrics.roc_auc_score(y_test, probs[1])
tree_confus_matrix = metrics.confusion_matrix(y_test, predicted)
tree_classification_report = metrics.classification_report(y_test, predicted)
tree_precision = metrics.precision_score(y_test, predicted, pos_label=1)
tree_recall = metrics.recall_score(y_test, predicted, pos_label=1)
tree_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

endpoint_name = f"{stack_name}-{commit_id[:7]}"

predictor = tree_model.deploy(
    initial_instance_count=1, instance_type="ml.m4.xlarge", endpoint_name=endpoint_name
)

