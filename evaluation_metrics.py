"""
This code provides a generalisation of classification reporst to multilabel classifications where a first a confusion
matrix is computed having all matches on the diagonal and taking all missmatching combination as off-diagonal entries.
From confusion matrix precision, recall and F1 score is straight-forward.
If the number of classes is large, it might make sense to take similarities between classes into account. This is simply
done by taking multiplying the confusion matrix with the similarity matrix (c.f. slide 18).
"""

import pandas as pd
import numpy as np

def get_onehot(class_idc, n):
    onehot = np.zeros(n)
    onehot[class_idc]=1
    return onehot

def get_confusion(df_data, classes, k, truth, predictions):
    """
    Computes confusion matrix which contains at C_ij the number of data points whith ground truth i 
    and prediction j.
    Parameters
    ----------
    :param df_test: dataframe containing test data
    :param classes: list of classes
    :param k: int, number of predicted labels
    :param truth: list, column name(s) of ground truth
    :param predictions: list, column name(s) of prediction 
    :return: returns confusion matrix C
    """
    df_test = pd.DataFrame()
    df_test["truth"] = df_data[truth].apply(set,axis =1).apply(lambda a: {x for x in a if x==x})
    df_test["predictions"] = df_data[predictions].apply(set, axis=1).apply(lambda a: {x for x in a if x==x})
    print("compute diffecence sets")

    #     handle multilabel classification by computing matching and missclassified labels separately
    for i in range(k):

        # remove matches from truth and predictions        
        df_test["truth_%i" %i] = df_test.apply(lambda x: x["truth"] - x["predictions"],
                                           axis =1).apply(list).str[i].dropna()
        df_test["predictions_%i" %i] = df_test.apply(lambda x: x["predictions"] - x["truth"],
                                           axis =1).apply(list).str[i].dropna()

    print("compute intersections")
    
    # identify matches     
    for i in range(k):
        df_test["matches_%i" %i] = df_test.apply(lambda x: x["predictions"] & x["truth"],
                                           axis =1).apply(list).str[i].dropna()

    truth_match = ["matches_%i" %i for i in range(k)]
    truth_nomatch = ["truth_%i" %i for i in range(k)]
    pred_nomatch = ["predictions_%i"%i for i in range(k)]
    n = len(classes)
    idx = pd.Series(index=classes, data=range(len(classes)))

    print("determine vectors")

    # compute one-hot encoded vectors y for each data point x as
    # x_t = i => y_t = (0, 0, ...,1,0,...0) with y havin all zeros only at index i 1.
    onehots_match = df_test.apply(
        lambda x: get_onehot(idx.get(x[truth_match].dropna(),[]),n), axis =1)

    onehots_nomatch_truth =  df_test.apply(
        lambda x: get_onehot(idx.get(x[truth_nomatch].dropna(), []),n), axis =1)
    onehots_nomatch_pred = df_test.apply(
        lambda x: get_onehot(idx.get(x[pred_nomatch].dropna(), []),n), axis =1)

    # We compute one-hot matrices for predicted and true classes as
    # Yˆpred = (yˆpred_1,..., yˆpred_T)
    # and Y_truth = (yˆtruth_1,..., yˆtruth_T)
    onehots_match=onehots_match.apply(pd.Series)
    onehots_nomatch_truth=onehots_nomatch_truth.apply(pd.Series)
    onehots_nomatch_pred=onehots_nomatch_pred.apply(pd.Series)

    onehots_nomatch_pred = onehots_nomatch_pred.fillna(0)
    onehots_nomatch_pred = ((onehots_nomatch_pred.T/onehots_nomatch_pred.sum(axis = 1))).fillna(0).T

    # compute confusion matrix as matris product from one-hot encoded label matrices  
    # C = Y_truth*Y_pred'
    confusion_nomatch = (onehots_nomatch_truth.T.dot(onehots_nomatch_pred)).T

    confusion_match = onehots_match.T.dot(onehots_match)
    confusion = confusion_nomatch + np.diag(np.diag(confusion_match))
    confusion.index = classes
    confusion.columns = classes
    return confusion

def get_report(confusion, similarity = None):
    """
    Computes classification report including precision, recall, f1-score and support per class.
    Parameters
    ----------
    :param confusion: dataframe, confusion matrix
    :param similarity: dataframe, similarity matrix
    :return: dataframe, classification report per class
    """
    
    if similarity is None:
        similarity = pd.DataFrame(np.eye(confusion.shape[0]), index=confusion.index,
                         columns=confusion.columns)
    cps=confusion.index
    precision=(pd.Series(np.diag(confusion.dot(similarity.loc[cps,cps])), 
                     index=cps, name="precision")/ confusion.sum(axis = 1)).fillna(0)
    recall=(pd.Series(np.diag(similarity.loc[cps,cps].dot(confusion)), 
                      index=cps, name="recall")/ confusion.sum(axis = 0)).fillna(0)
    support = confusion.sum()
    df_sim_report = precision.to_frame("precision").join(recall.to_frame("recall"))
    f1score = df_sim_report.precision*df_sim_report.recall*2/(df_sim_report.precision+df_sim_report.recall)
    df_sim_report=df_sim_report.join(f1score.to_frame("f1score")).join(support.to_frame("support"))
    return df_sim_report

def get_summary(df_sim_report, confusion):
    """
    Computes summary of classification report summary
    Parameters
    ----------
    :param df_sim_report: dataframe classificatin report per class
    :param confusion: dataframe, confusion matrix
    :return: returns macro average and weighted average or precision recall and f1-score
    """

    df_sim_report = df_sim_report[df_sim_report.support>0]
    sim_summary = pd.DataFrame(index=["macro avg","weighted avg"], 
                           columns=["precision","recall","f1-score","support"])
    sim_summary["support"] = df_sim_report.support.sum()

    sim_summary.loc["macro avg", "precision"] = df_sim_report.precision.mean()
    sim_summary.loc["macro avg", "recall"] = df_sim_report.recall.mean()

    sim_summary.loc["weighted avg", "precision"] = (df_sim_report.precision* 
                                                    df_sim_report.support).sum()/df_sim_report.support.sum()
    sim_summary.loc["weighted avg", "recall"] = (df_sim_report.recall* 
                                                 df_sim_report.support).sum()/df_sim_report.support.sum()

    sim_summary["f1-score"] = sim_summary.precision*sim_summary.recall*2/(sim_summary.precision
                                                                          +sim_summary.recall)
    return sim_summary
