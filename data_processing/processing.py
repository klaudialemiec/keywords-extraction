import xml.etree.ElementTree as ET
import configparser
import glob
import pandas as pd
import string
from data_preprocessing.tagger import lemmatize
from data_loader.data_loader import read_xml_file, load_text_data
from stop_words import get_stop_words
from typing import Any, Tuple


def process_data_IOB(xml_filename: str, keywords: str) -> Tuple:
    origin_text, base_text, ctag = read_xml_file(xml_filename)
    base_keywords = find_base_form(lemmatize(keywords))

    # words to lowercase
    origin_words_list = [x.lower() for x in origin_text.split()]
    base_words_list = [x.lower() for x in base_text.split()]
    base_keywords = [x.lower() for x in base_keywords.split(", ")]
    origin_keywords = [x.lower() for x in keywords.split(", ")]

    # remove punctuations
    origin_words_list = remove_punctuation(origin_words_list)
    base_words_list = remove_punctuation(base_words_list)
    origin_keywords = remove_punctuation(origin_keywords)

    origin_keywords_in_text = get_keywords_from_text(
        origin_keywords, origin_words_list)
    base_keywords_in_text = get_keywords_from_text(
        base_keywords, base_words_list)
    labels_base, labels_origin = labelling_texts(
        origin_words_list, base_words_list, base_keywords_in_text)

    return origin_text, base_text, keywords, base_keywords, base_keywords_in_text, origin_keywords_in_text, \
        base_words_list, origin_words_list, ctag, labels_base, labels_origin


def get_keywords_from_text(keywords: list, words_list: list) -> list:
    keywords_in_text = []
    for i in range(len(keywords)):
        if len(keywords[i].split()) > 1:
            keywords_tmp = ""
            for keyword in keywords[i].split():
                if keyword in words_list:
                    keywords_tmp += keyword + " "
            if keywords_tmp != "":
                keywords_in_text.append(keywords_tmp[:-1])
        elif keywords[i] in words_list:
            keywords_in_text.append(keywords[i])

    return keywords_in_text


def remove_punctuation(words_list: list) -> list:
    result = [''.join(w for w in words if w not in string.punctuation)
              for words in words_list]
    result = [w for w in result if w]
    return result


def find_base_form(lemmatize: Any) -> str:
    root = ET.fromstring(lemmatize)
    base = [word.find('lex').find('base').text for word in root.iter('tok')]
    base_keywords = ' '.join(base)
    return base_keywords.replace(" ,", ",")


def labelling_texts(original_text: list, base_text: list, base_keywords: str) -> Tuple[list, list]:
    labels_base = [labelling_word_IOB(word, base_keywords)
                   for word in base_text]
    labels_origin = [labelling_word_IOB(
        word, base_keywords) for word in original_text]
    return labels_base, labels_origin


def labelling_word_IOB(word: str, base_keywords: list) -> str:
    long_base_keywords = [
        base_word for base_word in base_keywords if " " in base_word]

    if word in base_keywords:
        return 'I'
    else:
        for item in long_base_keywords:
            splitted = item.split()
            if word in splitted:
                if splitted[0] == word:
                    return 'B'
                else:
                    return 'I'
    return 'O'
