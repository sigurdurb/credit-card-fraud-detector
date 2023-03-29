#!/usr/bin/env python
# coding: utf-8

# In[312]:


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# In[313]:


from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, auc, roc_auc_score, ConfusionMatrixDisplay

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    credidcard_fraud_df = pd.read_csv('../../creditcard.csv')

    y = credidcard_fraud_df['Class']
    X = credidcard_fraud_df
    X.pop('Class')

    X_train, X_validate, y_train, y_validate = train_test_split(X, y, train_size=0.8, test_size=0.2, shuffle=True,
                                                                random_state=37, stratify=y)
    return X_train, X_validate, y_train, y_validate

   


# In[314]:


def do_data( drop_cols):
    X_train, _, y_train, _ = load_data()
    
    # Local train and test set
    #X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=0.8, test_size=0.2)
    if drop_cols:
        df = X_train.drop(drop_cols,axis=1)
    else:
        #df = pd.concat([X_train[['Time','Amount']],less_extra(X_train)])
        df = X_train[['Time','Amount','V2', 'V3', 'V4', 'V6', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V19', 'V26', 'V27']]
    #df["Time"] = df["Time"] //60//60%24
    
    #print(df.head(3))
    #print(df.columns.values)
    # Split the dataset into features (X) and target (y)
    X = df# df.iloc[:, :-1] # Extract all columns except the last one (which is the target)
    #y = df.iloc[:, -1] # Extract only the last column (which is the target)
    
    # Split the dataset into training and testing sets
    return train_test_split(X, y_train, test_size=0.2, train_size=0.8)


# In[315]:


def predict(X_train, X_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    y_pred_prob = gnb.predict_proba(X_test)
    import pickle

    with open('nb_v2', 'wb') as f:
        pickle.dump(gnb, f)

    return y_pred, y_pred_prob, gnb


# In[316]:


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





# In[317]:


def run_detector(drop_cols = False):
    # Print the sizes of the training and testing sets
    import matplotlib.pyplot as plt
    
    X_train, X_test, y_train, y_test = do_data(drop_cols)
    print("Training set size:", len(X_train))
    print("Testing set size:", len(X_test))
    y_pred, y_pred_prob, gnb = predict(X_train, X_test, y_train, y_test)
    conf_mtx = print_performance(y_test,y_pred,y_pred_prob)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mtx, display_labels=gnb.classes_)
    disp.plot()
    plt.show()


# In[318]:


if __name__ == '__main__':
    drop_cols = None
    
    run_detector(drop_cols)
    #drop_cols = ['V2', 'V3', 'V4', 'V6', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V19', 'V26', 'V27'] # Couls also try V20 instead of V21
    #run_detector(drop_cols)
    #drop_cols = ['Time','V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8']
    #run_detector(drop_cols)
    #drop_cols = ['Time','V1','V20','V21','V7','V13','V26','V19','V24','V15','V13','V27','V28']
    #run_detector(drop_cols)
    


# In[ ]:





# In[ ]:





# In[ ]:




