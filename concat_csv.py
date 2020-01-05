import re, glob, sys, os, argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('csv_dir')
parser.add_argument('output_file')
args = parser.parse_args()

# 指定されたCSVディレクトリ配下の全CSVファイルのパス
files = glob.glob(args.csv_dir + '/*.csv')

major_class_name = "業種(大分類)_分類名"
medium_class_name = "業種(中分類)_分類名"
small_class_name = "業種(小分類)_分類名"

# 全てのCSVファイルを読み込み、分かち書きなど行ってoutputファイルに書き出す。
wakati_only_file = args.output_file.replace('.csv', '.txt')

with open(args.output_file, "w") as output_csv:
  output_csv.writelines('業種(大分類),文章\n')

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
        label_str = ''

        if type(major_class[col]) is str:
           label_str += str(major_class[col])

        nodes.append(label_str)            


        # 改行コードを含む場合があるため除去する
        sentence = ''.join(str(sentence).splitlines())
        nodes.append('"' + sentence + '"')

        output_csv.writelines(','.join(nodes) + '\n')

