from pycrfsuite import Trainer, Tagger, ItemSequence
from ast import literal_eval
import pandas as pd


def train(features: pd.Series, labels: pd.Series) -> None:
    trainer = Trainer(verbose=False)
    features = features.tolist()
    labels = labels.tolist()

    for idx in range(len(features)):
        trainer.append(ItemSequence(features[idx]), literal_eval(labels[idx]))
    trainer.train('crf.model')


def test(features: pd.Series) -> list:
    tagger = Tagger()
    tagger.open('crf.model')
    y_pred = [tagger.tag(xseq) for xseq in features]
    return y_pred
