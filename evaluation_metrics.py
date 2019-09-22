import pandas as pd
import numpy as np

def get_onehot(fields, classes):
    onehot = pd.Series(index = classes)
    onehot[fields.dropna()] = 1
    return onehot

def get_confusion(df_test, classes, k, truth, predictions):
    df_test["truth"] = df_test[truth].apply(set,axis =1).apply(lambda a: {x for x in a if x==x})
    df_test["predictions"] = df_test[predictions].apply(set, axis=1).apply(lambda a: {x for x in a if x==x})

    for i in range(k):
        df_test["truth_%i" %i] = df_test.apply(lambda x: x["truth"] - x["predictions"],
                                           axis =1).apply(list).str[i].dropna()
        df_test["predictions_%i" %i] = df_test.apply(lambda x: x["predictions"] - x["truth"],
                                           axis =1).apply(list).str[i].dropna()

    for i in range(k):
        df_test["matches_%i" %i] = df_test.apply(lambda x: x["predictions"] & x["truth"],
                                           axis =1).apply(list).str[i].dropna()

    truth_match = ["matches_%i" %i for i in range(k)]
    truth_nomatch = ["truth_%i" %i for i in range(k)]
    pred_nomatch = ["predictions_%i"%i for i in range(k)]

    onehots_match = df_test.apply(lambda x: get_onehot(x[truth_match], classes), axis=1).fillna(0)
    onehots_nomatch_truth = df_test.apply(lambda x: get_onehot(x[truth_nomatch], classes), axis=1).fillna(0)
    onehots_nomatch_pred = df_test.apply(lambda x: get_onehot(x[pred_nomatch], classes), axis=1).fillna(0)

    onehots_nomatch_pred = onehots_nomatch_pred.fillna(0)
    onehots_nomatch_pred = ((onehots_nomatch_pred.T/onehots_nomatch_pred.sum(axis = 1))).fillna(0).T

    confusion_nomatch = onehots_nomatch_truth.T.dot(onehots_nomatch_pred)

    confusion_match = onehots_match.T.dot(onehots_match)
    confusion = confusion_nomatch + np.diag(np.diag(confusion_match))
    return confusion

def get_report(confusion, S = None):
    
    if S is None:
        S = pd.DataFrame(np.eye(confusion.shape[0]), index=confusion.index,
                         columns=confusion.columns)
    cps=confusion.index
    precision=(pd.Series(np.diag(confusion.dot(S.loc[cps,cps])), 
                     index=cps, name="precision")/ confusion.sum(axis = 1)).fillna(0)
    recall=(pd.Series(np.diag(S.loc[cps,cps].dot(confusion)), 
                      index=cps, name="recall")/ confusion.sum(axis = 0)).fillna(0)
    support = confusion.sum()
    df_sim_report = precision.to_frame("precision").join(recall.to_frame("recall"))
    f1score = df_sim_report.precision*df_sim_report.recall*2/(df_sim_report.precision+df_sim_report.recall)
    df_sim_report=df_sim_report.join(f1score.to_frame("f1score")).join(support.to_frame("support"))
    return df_sim_report
#    df_sim_report.index = df_cpids.set_index("id").name[df_sim_report.index]

def get_summary(df_sim_report, confusion, S = None):
    
    cps=confusion.index
    if S is None:
        S = pd.DataFrame(np.eye(confusion.shape[0]), index=cps,
                     columns=cps)
    df_sim_report = df_sim_report[df_sim_report.support>0]
    sim_summary = pd.DataFrame(index=["micro avg","macro avg","weighted avg"], 
                           columns=["precision","recall","f1-score","support"])
    sim_summary["support"] = df_sim_report.support.sum()
    sim_summary.loc["micro avg", "precision"] = pd.Series(np.diag(confusion.dot(S.loc[cps,cps])), 
                                 index=cps, name="precision").sum()/confusion.sum().sum()
    sim_summary.loc["micro avg", "recall"] = pd.Series(np.diag(S.loc[cps,cps].dot(confusion)), 
                                 index=cps, name="recall").sum()/confusion.sum().sum()

    sim_summary.loc["macro avg", "precision"] = df_sim_report.precision.mean()
    sim_summary.loc["macro avg", "recall"] = df_sim_report.recall.mean()

    sim_summary.loc["weighted avg", "precision"] = (df_sim_report.precision* 
                                                    df_sim_report.support).sum()/df_sim_report.support.sum()
    sim_summary.loc["weighted avg", "recall"] = (df_sim_report.recall* 
                                                 df_sim_report.support).sum()/df_sim_report.support.sum()

    sim_summary["f1-score"] = sim_summary.precision*sim_summary.recall*2/(sim_summary.precision
                                                                          +sim_summary.recall)
    return sim_summary
