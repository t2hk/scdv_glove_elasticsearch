import re, glob, sys, os, argparse
import pandas as pd
import numpy as np

from sudachipy import tokenizer
from sudachipy import dictionary

parser = argparse.ArgumentParser()
parser.add_argument('csv_dir')
parser.add_argument('output_file')
parser.add_argument('need_category')
args = parser.parse_args()

# 指定されたCSVディレクトリ配下の全CSVファイルのパス
files = glob.glob(args.csv_dir + '/*.csv')

tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C

major_class_name = "業種(大分類)_分類名"
medium_class_name = "業種(中分類)_分類名"
small_class_name = "業種(小分類)_分類名"

# ストップワードの読み込み
stopwords = []
with open('stopwords.txt') as sw:
    stopwords = sw.read().splitlines()

# 全てのCSVファイルを読み込み、分かち書きなど行ってoutputファイルに書き出す。
wakati_only_file = args.output_file.replace('.csv', '.txt')

with open(args.output_file, "w") as output_csv:
 with open(wakati_only_file, "w") as output_txt:
  output_csv.writelines('業種(大分類),文章,分かち書き\n')

  for file in files:
    csv_file_name = os.path.basename(file)

    df = pd.read_csv(file)

    # 状況の列名をファイルの種類ごとに判断する。
    col_name = '災害状況'
    if 'kikaisaigai' in file:
        col_name = '災害発生状況'

    print(file)
    sentences = df[col_name]

    # 分類の列
    major_class = df[major_class_name]
    medium_class = df[medium_class_name]
    small_class = df[small_class_name]

    for col, sentence in enumerate(sentences):
        nodes = []
        
        # カテゴリー指定の場合、カテゴリを読み込んで設定する。
        if args.need_category.upper() == 'TRUE':
            label_str = ''

            if type(major_class[col]) is str:
              label_str += str(major_class[col])

            ''' 
            if type(medium_class[col]) is str:
              label_str += '__label__' + str(medium_class[col]) + ','

            if type(small_class[col]) is str:
              label_str += '__label__' + str(small_class[col]) + ','
            ''' 
            nodes.append(label_str)            


        # 改行コードを含む場合があるため除去する
        sentence = ''.join(str(sentence).splitlines())
        nodes.append('"' + sentence + '"')

        # Suachiで数字以外の名詞、動詞、形容詞のみを分かち書きする。
        # 単語は原型を取得する。
        tokens = []
        ms = tokenizer_obj.tokenize(sentence, mode)
        for m in ms:
            #form = m.dictionary_form()
            form = m.surface()

            form = form.strip()

            # ストップワードの場合、スキップする。
            if form in stopwords:
                continue
            
            # 一文字、かつ、半角英数記号の場合は対象外とする。
            if len(form) == 1 and re.search('[\u0000-\u007F]+', form):
                continue

            pos = m.part_of_speech()
            #nodes.append(form)
            if pos[0] in ["名詞", "動詞", "形容詞"] and pos[1] != "数詞":
                #print("{} {}".format(form, pos[0]))
                tokens.append(form)

        output_csv.writelines(','.join(nodes) + ',"' + ' '.join(tokens) + '"\n')
        output_txt.writelines(' '.join(tokens) + '\n')

