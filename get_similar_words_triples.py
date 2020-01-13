from gensim.models import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import pandas as pd

glove_vector_file = "vectors.txt"
gensim_glove_vector_file = "gensim_glove_vectors.txt"
top_k = 10

words_triple_file = 'similarity_words.ttl'

# GloVeの単語ベクトルファイルを読み込み、単語数とベクトルサイズを付与した処理用のファイルを作成する。
vectors = pd.read_csv(glove_vector_file, delimiter=' ', index_col=0, header=None)

vocab_count = vectors.shape[0]  # 単語数
num_features = vectors.shape[1]  # 次元数

print("単語数：{}  次元数：{}".format(vocab_count, num_features))

glove_vectors = KeyedVectors.load_word2vec_format(gensim_glove_vector_file, binary=False)
words = list(glove_vectors.vocab.keys())

sim_words_list = []

with open(words_triple_file, 'w') as f:
  for word in words:
    sim_words = glove_vectors.most_similar(word, [], top_k)
    
    for sim_word in sim_words:
        triple = '"{}" owl:equivalentClass "{}"'.format(word, sim_word[0])
    
        sim_words_list.append(triple)
        f.writelines(triple + '\n')


len(sim_words_list)
