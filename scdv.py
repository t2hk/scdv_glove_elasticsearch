import logging, argparse, pickle, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from gensim.models import KeyedVectors
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class SparseCompositeDocumentVectors:
    def __init__(self, num_clusters,  pname1, pname2):
        self.min_no = 0
        self.max_no = 0
        self.prob_wordvecs = {}
        
        #### 読み込むファイルの設定
        # GloVeの単語ベクトルファイル
        self.glove_word_vector_file = "../elasticsearch/es_glove_vectors.txt"
        #self.glove_word_vector_file = "../glove/glove_vectors.txt"

        #### 出力するファイルの設定
        # GloVeの単語ベクトルに単語数とベクトルサイズを付与したファイル
        self.gensim_glove_word_vector_file = "../elasticsearch/es_gensim_glove_vectors.txt"
        #self.gensim_glove_word_vector_file = "../glove/gensim_glove_vectors.txt"
        
        # GMMの結果を保存するPickleファイル
        self.pname1 = pname1
        self.pname2 = pname2

        #### その他パラメータ
        # GMMのクラスタ数
        self.num_clusters = num_clusters
        
        # GloVeの次元数
        self.num_features = 50
    
    def load_glove_vector(self):
        # GloVeの単語ベクトルファイルを読み込み、単語数とベクトルサイズを付与した処理用のファイルを作成する。
        vectors = pd.read_csv(self.glove_word_vector_file, delimiter=' ', index_col=0, header=None)
        
        vocab_count = vectors.shape[0]  # 単語数
        self.num_features = vectors.shape[1]  # 次元数

        with open(self.glove_word_vector_file, 'r') as original, open(self.gensim_glove_word_vector_file, 'w') as transformed:
            transformed.write(f'{vocab_count} {self.num_features}\n')
            transformed.write(original.read())  # 2行目以降はそのまま出力

        # GloVeの単語ベクトルを読み込む
        self.glove_vectors = KeyedVectors.load_word2vec_format(self.gensim_glove_word_vector_file, binary=False)

    def cluster_GMM2(self):   
        glove_vectors = self.glove_vectors.vectors
        
        # Initalize a GMM object and use it for clustering.
        gmm_model = GaussianMixture(n_components=num_clusters, covariance_type="tied", init_params='kmeans', max_iter=100)
        # Get cluster assignments.
        gmm_model.fit(glove_vectors)
        idx = gmm_model.predict(glove_vectors)
        print ("Clustering Done...")
        # Get probabilities of cluster assignments.
        idx_proba = gmm_model.predict_proba(glove_vectors)
        # Dump cluster assignments and probability of cluster assignments. 
        pickle.dump(idx, open(self.pname1,"wb"))
        print ("Cluster Assignments Saved...")

        pickle.dump(idx_proba,open(self.pname2, "wb"))
        print ("Probabilities of Cluster Assignments Saved...")
        return (idx, idx_proba)        
        
    def cluster_GMM(self):
        # GMMによるクラスタリング
        
        clf = GaussianMixture(
            n_components=self.num_clusters,
            covariance_type="tied",
            init_params="kmeans",
            max_iter=50
        )
        
        glove_vectors = self.glove_vectors.vectors
        # Get cluster assignments.
        clf.fit(glove_vectors)
        idx = clf.predict(glove_vectors)
        print("Clustering Done...")
        # Get probabilities of cluster assignments.
        idx_proba = clf.predict_proba(glove_vectors)
        # Dump cluster assignments and probability of cluster assignments.
        pickle.dump(idx, open(self.pname1, "wb"))
        print("Cluster Assignments Saved...")
        pickle.dump(idx_proba, open(self.pname2, "wb"))
        print("Probabilities of Cluster Assignments saved...")
        return (idx, idx_proba)

    def read_GMM(self):
        # GMMモデルを読み込む。
        
        idx = pickle.load(open(self.idx_name, "rb"))
        idx_proba = pickle.load(open(self.idx_proba_name, "rb"))
        print("Cluster Model Loaded...")
        return (idx, idx_proba)

    def get_idf_dict(self, corpus):
        # IDFを算出する。
        # corpus : 分かち書きした文章のリスト
        
        # 単語の数をカウントする
        count_vectorizer = CountVectorizer()
        X_count = count_vectorizer.fit_transform(corpus)

        # scikit-learn の TF-IDF 実装
        tfidf_vectorizer = TfidfVectorizer(token_pattern="(?u)\\b\\w+\\b")
        X_tfidf = tfidf_vectorizer.fit_transform(corpus)

        feature_names = tfidf_vectorizer.get_feature_names()
        idf = tfidf_vectorizer.idf_

        word_idf_dict = {}
        for pair in zip(feature_names, idf):
            word_idf_dict[pair[0]] = pair[1]
        
        return feature_names, word_idf_dict

    def get_probability_word_vectors(self, corpus):
        """
        corpus: 分かち書き済みの文章のリスト
        """
        
        # GloVeの単語ベクトルを読み込む。
        self.load_glove_vector()
        
        # 単語毎のGMMクラスタの確率ベクトル
        idx, idx_proba = self.cluster_GMM()
 
        # 各単語が属する確率が高いクラスタのインデックス
        word_centroid_map = dict(zip(self.glove_vectors.index2word, idx))
        # 各単語が、各クラスタに属する確率
        word_centroid_prob_map = dict(zip(self.glove_vectors.index2word, idx_proba))     
        
        # TF-IDFを算出する。
        featurenames, word_idf_dict = self.get_idf_dict(corpus)
        
        for word in word_centroid_map:
            self.prob_wordvecs[word] = np.zeros(self.num_clusters * self.num_features, dtype="float32")
            for index in range(self.num_clusters):
                try:
                    self.prob_wordvecs[word][index*self.num_features:(index+1)*self.num_features] = \
                        self.glove_vectors[word] * word_centroid_prob_map[word][index] * word_idf_dict[word]
                except:
                    continue
        self.word_centroid_map = word_centroid_map

    def create_cluster_vector_and_gwbowv(self, tokens, flag):
        # SDV(Sparse Document Vector)を組み立てる。
        
        bag_of_centroids = np.zeros(self.num_clusters * self.num_features, dtype="float32")
        for token in tokens:
            try:
                temp = self.word_centroid_map[token]
            except:
                continue
            bag_of_centroids += self.prob_wordvecs[token]
        norm = np.sqrt(np.einsum('...i,...i', bag_of_centroids, bag_of_centroids))
        if norm != 0:
            bag_of_centroids /= norm
            
        # 訓練で作成したベクトルをスパース化するために最小と最大を記録しておく。
        if flag:
            self.min_no += min(bag_of_centroids)
            self.max_no += max(bag_of_centroids)
        return bag_of_centroids

    def make_gwbowv(self, corpus, train=True):
        # ドキュメントベクトルのマトリクスを作成する。
        # gwbowvには通常のドキュメントベクトルが格納される。
        gwbowv = np.zeros((len(corpus), self.num_clusters*self.num_features)).astype(np.float32)
        cnt = 0
        for tokens in tqdm(corpus):
            gwbowv[cnt] = self.create_cluster_vector_and_gwbowv(tokens, train)
            cnt += 1

        return gwbowv

    def dump_gwbowv(self, gwbowv, path="gwbowv_matrix.npy", percentage=0.04):
        # スパース化したドキュメントベクトルを保存する。
        
        # スパース化するための閾値を算出する。
        min_no = self.min_no*1.0/gwbowv.shape[0]
        max_no = self.max_no*1.0/gwbowv.shape[0]
        print("Average min: ", min_no)
        print("Average max: ", max_no)
        thres = (abs(max_no) + abs(min_no))/2
        thres = thres * percentage
        
        # 閾値未満のベクトルを0とし、スパース化する。
        temp = abs(gwbowv) < thres
        gwbowv[temp] = 0
        np.save(path, gwbowv)
        print("SDV created and dumped...")

    def load_matrix(self, name):
        return np.load(name)

def parse_args():
    parser = argparse.ArgumentParser(
        description="GloVeとSCDVのパラメータの設定"
    )
    parser.add_argument('--csv_file', type=str)
    parser.add_argument(
        '--num_clusters', type=int, default=20
    )
    parser.add_argument(
        '--pname1', type=str, default="gmm_cluster.pkl"
    )
    parser.add_argument(
        '--pname2', type=str, default="gmm_prob_cluster.pkl"
    )

    return parser.parse_args()

def build_model(csv_file, num_clusters, gmm_pname1, gmm_pname2):
    df = pd.read_csv(csv_file)

    tokens = df['分かち書き']
    categories = df['業種(大分類)']
    index = df['ID']

    vec = SparseCompositeDocumentVectors(num_clusters, gmm_pname1, gmm_pname2)
    # 確率重み付き単語ベクトルを求める
    vec.get_probability_word_vectors(tokens)
    # データからSCDVを求める
    gwbowv = vec.make_gwbowv(tokens)

    print("id len:{}, gwbowv len:{}".format(len(index), len(gwbowv)))

    return zip(index, gwbowv)

def main(args):
    df = pd.read_csv(args.csv_file)
    categories = df['業種(大分類)'].unique()
    NUM_TOPICS = len(categories)

    # 訓練データとtestデータに分ける
    train_data, test_data, train_label, test_label, train_id, test_id = train_test_split(
        df['分かち書き'], df['業種(大分類)'], df['ID'],
        test_size=0.1, train_size=0.9, stratify=df['業種(大分類)'], shuffle=True)

    vec = SparseCompositeDocumentVectors(args.num_clusters, args.pname1, args.pname2)
    # 確率重み付き単語ベクトルを求める
    vec.get_probability_word_vectors(train_data)
    # 訓練データからSCDVを求める
    train_gwbowv = vec.make_gwbowv(train_data)
    # テストデータからSCDVを求める
    test_gwbowv = vec.make_gwbowv(test_data, False)

    print("train size:{}  vector size:{}".format(len(train_gwbowv), len(train_gwbowv[0])))
    print("test size:{}  vector size:{}".format(len(test_gwbowv), len(test_gwbowv[0])))

    print("Test start...")

    start = time.time()
    clf = lgb.LGBMClassifier(objective="multiclass")
    clf.fit(train_gwbowv, train_label)
    test_pred = clf.predict(test_gwbowv)

    # print(test_pred)

    print ("Report")
    print (classification_report(test_label, test_pred, digits=6))
    print ("Accuracy: ",clf.score(test_gwbowv, test_label))
    print ("Time taken:", time.time() - start, "\n")

if __name__ == "__main__":
    main(parse_args())
