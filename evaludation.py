import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from load_data import load_data
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef, cohen_kappa_score, auc
from sklearn.metrics import confusion_matrix

# ! pip install tabulate
from tabulate import tabulate

from random_forest_detector.random_forest_detector import predict as rf_predict
from deep_learning_detector.detector import predict as dnn_predict 
from naive_bayes_detector.nb_detector import predict as nb_predict


def get_stats_of_results(y_gt, y_pred):
    '''
    Given the ground truth labels (y_gt) and the predicted labels (y_pred),
    returns some basic stats on the performance:
    - Accuracy
    - AUC
    - Precision
    - Recall
    - F1-score
    - Matthews correlation coefficient
    - Cohen's kappa
    '''
    acc= accuracy_score(y_gt, y_pred)
    auc= accuracy_score(y_gt, y_pred)
    prec= precision_score(y_gt, y_pred)
    recall= recall_score(y_gt, y_pred)
    f1= f1_score(y_gt, y_pred)
    mcc= matthews_corrcoef(y_gt, y_pred)
    kappa= cohen_kappa_score(y_gt, y_pred)
    
    return acc, auc, prec, recall, f1, mcc, kappa

def baggings(y0,y1,y2):
    '''
    Returns a three new arrays following diffirent bagging stradegies
        - If every one agree, then agree
        - If majority agrees 
        - If majority agrees and y2 agrees with majority
    '''
    # Everyone agrees
    y3= np.logical_and.reduce([y0,y1,y2]).astype(int)
    
    # Majoridy roles
    y4= (np.sum([y0,y1,y2], axis=0) >= 2).astype(int)

    # Random Forest dominates
    y5= (np.logical_or(np.logical_and(y1, y0), np.logical_and(y1, y2))).astype(int)
    
    return y3, y4, y5


def calculate_revenue(X: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray) -> float:
    def map_to_revenue(row) -> float:
        match row.tolist():
            # Not fraud and not classified as fraud
            case [0, 0, amount]:
                return amount
            # Not fraud but misclassified as fraud
            case [0, 1, _]:
                return 0
            # Fraud but misclassified as not fraud
            case [1, 0, amount]:
                return -amount
            # Fraud that is correctly classified as fraud
            case [1, 1, _]:
                return 0

    data = np.dstack((y_true.to_numpy(), y_pred, X['Amount'].to_numpy()))
    return np.apply_along_axis(map_to_revenue, 2, data).sum()


def main():
    
    # Load data 
    _, X_eval, _, y_eval = load_data()

    # Stats we get
    stats_name= ["Acc", "AUC", "Prec", "Recall", "F1", "MCC", "Kappa"]

    # Predict
    y_pred_dnn= dnn_predict(X_eval)
    print(y_pred_dnn)
    dnn_stat = get_stats_of_results(y_eval, y_pred_dnn)

    y_pred_rf= rf_predict(X_eval)
    rf_stat = get_stats_of_results(y_eval, y_pred_rf)

    y_pred_nb= nb_predict(X_eval)
    nb_stat = get_stats_of_results(y_eval, y_pred_nb)

    y3, y4, y5 = baggings(y_pred_dnn, y_pred_rf, y_pred_nb)
    y3_stat, y4_stat, y5_stat = get_stats_of_results(y_eval, y3), get_stats_of_results(y_eval, y4), get_stats_of_results(y_eval, y5)
    
    # Print performance stats
    print(tabulate([["Deep Nural Network"]+list(dnn_stat), 
                    ["Random Forest"]+list(rf_stat), 
                    ["Naive Bayes"]+list(nb_stat),
                    ["Bagging All"]+list(y3_stat),
                    ["Bagging Major"]+list(y4_stat),
                    ["Bagging RF"]+list(y5_stat)], 
                    headers=["Detectors Name"]+stats_name, 
                    tablefmt="grid"))

    # Print revenue
    print(tabulate([["Deep Neural Network", f'€ {calculate_revenue(X_eval, y_eval, y_pred_dnn)}'],
                    ["Random Forest", f'€ {calculate_revenue(X_eval, y_eval, y_pred_rf)}'],
                    ["Naive Bayes", f'€ {calculate_revenue(X_eval, y_eval, y_pred_nb)}'],
                    ["Bagging All", f'€ {calculate_revenue(X_eval, y_eval, y3)}'],
                    ["Bagging Major", f'€ {calculate_revenue(X_eval, y_eval, y4)}'],
                    ["Bagging RF", f'€ {calculate_revenue(X_eval, y_eval, y5)}']],
                   headers=["Detectors Name", "Revenue"],
                   tablefmt="grid",
                   floatfmt=".2f"))

    # print confusion matiices
    for y_pred in [y_pred_dnn, y_pred_rf, y_pred_nb, y3, y4, y5]:
        conf_matrix = confusion_matrix(y_eval, y_pred)

        plt.figure(figsize=(5,5))
        # sns.heatmap(conf_matrix, annot=True, fmt="d");


        fig, ax = plt.subplots(figsize=(5, 5))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(2):
            for j in range(2):
                ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        
        plt.xlabel('Predictions', fontsize=14)
        plt.ylabel('Ground Truth', fontsize=14)
        plt.title('Confusion Matrix', fontsize=14)
        plt.show()


if __name__ == '__main__':
    main()
    