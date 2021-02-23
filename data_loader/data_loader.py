import xml.etree.ElementTree as ET
import glob
import pandas as pd
from typing import Tuple


def load_data(glob_pattern: str) -> pd.DataFrame:
    files = glob.glob(glob_pattern)
    df_list = [pd.read_csv(file, engine='python', encoding='utf-8')
               for file in files]
    return pd.concat(df_list)


def load_text_data(file: str) -> Tuple[str, str]:
    keywords = ""
    text = ""
    with open(file, 'r', encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == 0:
                keywords = line.rstrip()
            else:
                text += line.rstrip()
    return keywords, text


def read_xml_file(xml_filename: str) -> Tuple[str, str, list]:
    with open(xml_filename, 'r', encoding="utf-8") as content:
        tree = ET.parse(content)
        root = tree.getroot()

        origin = []
        base = []
        features = []
        for word in root.iter('tok'):
            origin.append(word.find('orth').text)
            base.append(word.find('lex').find('base').text)
            features.append(word.find('lex').find('ctag').text)

        origin_text = ' '.join(origin)
        base_text = ' '.join(base)

    return origin_text, base_text, features
