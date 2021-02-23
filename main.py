from data_loader.data_loader import load_data
from model import CRF
from model import evaluator
from feature_extraction import create_features_list
from ast import literal_eval
import numpy as np
from sklearn.model_selection import train_test_split


def get_keywords_from_labels(words, labels):
    keywords = []
    for words_row, labels_row in zip(words, labels):
        keywords_tmp = []
        words_row = literal_eval(words_row)

        for idx in range(len(words_row)):
            if labels_row[idx] == "B" and idx+1 < len(words_row) and labels_row[idx+1] == "I":
                keywords_tmp.append(words_row[idx] + " " + words_row[idx + 1])
            elif labels_row[idx] == "I":
                if idx > 0 and labels_row[idx-1] == "B":
                    continue
                else:
                    keywords_tmp.append(words_row[idx])
        keywords.append(keywords_tmp)
    return keywords


if __name__ == "__main__":
    dataset = load_data(".\\kpwr-1.1\\*\\result.csv")
    features = create_features_list(dataset)
    dataset["features"] = features

    train, test = train_test_split(dataset)
    CRF.train(train['features'], train['label_base'])
    preds = CRF.test(test['features'])

    keywords_true = test['base_keywords_in_text']
    keywords_pred = get_keywords_from_labels(test['base_words_list'], preds)

    prec_h, rec_h, f1_h = evaluator.hard_evaluation(
        keywords_true, keywords_pred)
    prec_s, rec_s, f1_s = evaluator.soft_evaluation(
        keywords_true, keywords_pred)

    print(
        f"Soft evalution: Precission: {np.mean(prec_s)*100}, Recall: {np.mean(rec_s)*100}, F1Score: {np.mean(f1_s)*100}")

    print(
        f"Hard evalution: Precission: {np.mean(prec_h)*100}, Recall: {np.mean(rec_h)*100}, F1Score: {np.mean(f1_h)*100}")

    for kt, kp in zip(keywords_true, keywords_pred):
        print("Keywords true: " + str(kt))
        print("Predicted keywords: " + str(kp))
        print("\n")
