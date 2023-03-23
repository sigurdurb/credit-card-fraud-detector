#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# In[37]:


from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, auc, roc_auc_score, ConfusionMatrixDisplay


# In[38]:


def load_data(test_size, drop_cols):
    df = pd.read_csv('../../creditcard.csv')
    if drop_cols:
        df = df.drop(drop_cols,axis=1)
    
    print(df.head(3))
    # Split the dataset into features (X) and target (y)
    X = df.iloc[:, :-1] # Extract all columns except the last one (which is the target)
    y = df.iloc[:, -1] # Extract only the last column (which is the target)


    # Split the dataset into training and testing sets
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


# In[39]:


def predict(X_train, X_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    y_pred_prob = gnb.predict_proba(X_test)

    return y_pred, y_pred_prob, gnb


# In[40]:


def print_performance(y_test, y_pred, y_pred_prob):
    print("Accuracy: ", accuracy_score(y_test,y_pred))
    print("Recall: ", recall_score(y_test,y_pred))
    print("Precision: ", precision_score(y_test,y_pred))
    print("F1: ", f1_score(y_test,y_pred))
    print("MCC: ", matthews_corrcoef(y_test,y_pred))
    print("ROC AUC: {}".format(roc_auc_score(y_test, y_pred_prob[:,1])))
    return confusion_matrix(y_test,y_pred)
    #print('test-set confusion matrix:\n', confusion_matrix(y_test,y_pred)) 
    


# In[ ]:





# In[41]:


def run_detector(drop_cols = False):
    # Print the sizes of the training and testing sets
    import matplotlib.pyplot as plt
    test_size = 0.2
    X_train, X_test, y_train, y_test = load_data(test_size, drop_cols)
    print("Training set size:", len(X_train))
    print("Testing set size:", len(X_test))
    y_pred, y_pred_prob, gnb = predict(X_train, X_test, y_train, y_test)
    conf_mtx = print_performance(y_test,y_pred,y_pred_prob)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mtx, display_labels=gnb.classes_)
    disp.plot()
    plt.show()


# In[43]:


if __name__ == '__main__':
    drop_cols = None
    
    run_detector(drop_cols)
    drop_cols = ['Time','V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8']
    run_detector(drop_cols)
    drop_cols = ['Time','V1','V20','V21','V7','V13','V26','V19','V24','V15','V13','V27','V28']
    run_detector(drop_cols)
    


# In[ ]:





# In[ ]:




