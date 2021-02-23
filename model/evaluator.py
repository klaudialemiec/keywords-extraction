from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from ast import literal_eval
import pandas as pd
from typing import Tuple


LABELS = {'I': 0, 'O': 1, 'B': 2, 'S': 3}


def label_data(data: list) -> list:
    for row in data:
        try:
            result = [LABELS[x] for x in literal_eval(row)]
        except:
            result = [LABELS[x] for x in row]
    return result


def hard_evaluation(keywords_true: pd.Series, keywords_pred: pd.Series) -> Tuple[float, float, float]:
    TP = 0.0
    FP = 0.0
    FN = 0.0

    for keywords_true_row, keywords_pred_row in zip(keywords_true, keywords_pred):
        keywords_true_row = literal_eval(keywords_true_row)
        for k_pred in keywords_pred_row:
            if k_pred in keywords_true_row:
                TP += 1
                keywords_true_row.remove(k_pred)
            else:
                FP += 1
        FN += len(keywords_true_row)

    if (TP + FP) == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if (TP + FN) == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    if (2 * TP + FP + FN) == 0:
        f_score = 0
    else:
        f_score = 2 * TP / (2 * TP + FP + FN)
    return precision, recall, f_score


def soft_evaluation(keywords_true: pd.Series, keywords_pred: pd.Series) -> Tuple[float, float, float]:
    TP = 0.0
    FP = 0.0
    FN = 0.0

    for keywords_true_row, keywords_pred_row in zip(keywords_true, keywords_pred):
        keywords_true_row = literal_eval(keywords_true_row)
        keywords_true_row_splitted = [
            x for i in keywords_true_row for x in i.split()]
        keywords_pred_row_splitted = [
            x for i in keywords_pred_row for x in i.split()]

        for k_pred in keywords_pred_row_splitted:
            if k_pred in keywords_true_row_splitted:
                TP += 1
                keywords_true_row_splitted.remove(k_pred)
            elif k_pred in keywords_true_row_splitted:
                TP += 1
                FP += 1
            else:
                FP += 1
        FN += len(keywords_true_row_splitted)

    if (TP + FP) == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if (TP + FN) == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    if (2 * TP + FP + FN) == 0:
        f_score = 0
    else:
        f_score = 2 * TP / (2 * TP + FP + FN)
    return precision, recall, f_score
