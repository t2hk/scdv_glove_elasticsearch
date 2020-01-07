# 概要
SCDVのGloVe版で文書を学習し、以下を行う。また、LDAによるトピックモデルの構築も行う。
  - SCDVによる類似文書の抽出
  - GloVeによる類似単語の抽出
  - LDAによるトピック抽出とWordCloudによるビジュアル化

文書データはElasticsearchに登録する。類似文書の検索はElasticsearchのText similarity searchを使用する。

# 環境

| 環境 | バージョン |
| --- | --- |
| OS | Ubuntu 18.04.3 LTS |
| Java | OpenJDK1.8.0_232 |
| Python | 3.6.9 |
| Elasticsearch | 7.5.1 |
| Kibana | 7.5.1 |
| Jupyter Lab | 1.2.4 |

# 環境構築手順

  * OpenJDKインストール
  
    ```
     $ sudo apt install openjdk-8-jdk -y
    ```

  * Python3 pipインストール
  
    ```
    $ sudo apt install python3-pip -y
    $ sudo pip3 install --upgrade pip
    ```

  * 各種ツールのインストール

    ```
    $ sudo apt install git -y
    $ sudo apt install maven -y
    $ sudo apt install fonts-ipaexfont
    ```

  * Jupyter labインストールと設定

    ```
    $ sudo pip3 install jupyterlab
    ```
    
    ログインし直して、設定を続ける。
  
    ```
    $ jupyter lab --generate-config
    Writing default config to: /home/[ユーザ]/.jupyter/jupyter_notebook_config.py

    $ vi /home/[ユーザ]/.jupyter/jupyter_notebook_config.py
      c.NotebookApp.allow_remote_access = True
      c.NotebookApp.ip = '[ホスト名/IPアドレス]'
    ```

  * Jupyter labの起動
  
    ```
    $ jupyter lab
    ```
    
  * Jupyter labへのアクセス
    起動時にコンソールに表示されるURLにブラウザでアクセスする。

  * Elasticsearch、Kibana、Sudachiのインストール
    以下の手順通りに実行する。
    https://qiita.com/t2hk/items/a5b647f4ca764b073a47

  * Pythonモジュールのインストール

    ```
    $ sudo pip3 install pandas numpy matplotlib gensim elasticsearch scikit-learn wordcloud tqdm sudachipy xlrd lightgbm
    ```
    
# 本プロジェクトの実行方法
  * プロジェクトの取得

    ```
    $ git clone https://github.com/t2hk/scdv_glove_elasticsearch.git
    $ cd scdv_glove_elasticsearch
    ```

  * データのダウンロード

    ```
    $ ./get_doc.sh
    $ mkdir excel_files
    $ mv *.xls* ./excel_files
    ```

  * ダウンロードしたデータをCSVに変換

    ```
    $ mkdir csv
    $ python3 excel_to_csv.py ./excel_files ./csv
    ```

  * CSVファイルを1つに結合する。

    ```
    $ python3 concat_csv.py ./csv accident_data.csv
    ```

  * 結合したCSVファイルをElasticsearchにロードする。
    - 結合したCSVファイルをダウンロードする。TeratermのSSH SCPなどを使用する。
    - ブラウザでKibanaにアクセスする（http://[Elasticsearchホスト]:5601)
    - 左のツールバーから「Machine Learning」を選択、「Import Data」の「Upload file」ボタンをクリックする。
    - 「Select or drag and drop a file」で、結合したCSVファイルを指定する。
    - データの内容が表示される。左下の「Import」ボタンをクリックする。
    - 「Index name」に「accident_data」と入力する。
    - 「Advanced」タブをクリックする。
    - 「Mappings」に以下を入力する。

        ```
        {
          "文章": {
            "type": "text"
          },
          "業種(大分類)": {
            "type": "keyword"
          },
          "scdv_vector" : {
                  "type" : "dense_vector",
                  "dims" : 1000
          }
        }
        ```

    - 「Index settings」に以下を入力する。

        ```
        {
          "number_of_shards": "1",
          "analysis": {
            "filter":{
              "my_posfilter" : {
                "type" : "sudachi_part_of_speech",
                "stoptags" : [
                  "接続詞",
                  "助動詞",
                  "助詞",
                  "記号",
                  "補助記号"
                ]
              }
            },
            "analyzer": {
              "sudachi_analyzer": {
                "filter": [
                  "sudachi_baseform",
                  "lowercase",
                  "my_posfilter"
                ],
                "type": "custom",
                  "tokenizer": "sudachi_tokenizer"
                }
              },
              "tokenizer": {
                "sudachi_tokenizer": {
                  "mode": "search",
                  "type": "sudachi_tokenizer",
                  "discard_punctuation": "true",
                  "resources_path": "/opt/elasticsearch/config/sudachi_tokenizer/",
                  "settings_path": "/opt/elasticsearch/config/sudachi_tokenizer/sudachi_fulldict.json"
                }
              }
            }
          }
        ```

    - 「Import」ボタンを押下する。インポートされる。

  * Elasticsearchでトークナイズし、結果を保存する。

    ```
    $ python3 es_tokenize.py --host [Elasticsearchホスト] --index accident_data --output es_accident_data.csv
    ```
    
    以下のファイルが作成される。
    - Elasticsearchで分かち書きした結果とIDを含むCSVファイル
    - 分かち書きした結果のみのTXTファイル

  * GloVeで単語を学習する。
    - Gloveの入手
    
      ```
      $ cd ..
      $ git clone https://github.com/stanfordnlp/GloVe.git
      $ cd GloVe
      $ make
      ```

    - 実行コマンドをコピーする。
    
      ```
      $ cp ../scdv_glove_elasticsearch/run_glove.sh ./
      ```
    
    - run_glove.shの設定を編集する。
      基本的に「DATA_DIR」に「scdv_glove_elasticsearch」のディレクトリパスを記述すれば良い。

  * GloVeで単語を学習する。
    
    ```
    $ ./run_glove.sh
    ```
  
  「scdv_glove_elasticseach」ディレクトリにモデルデータが作成される。

  * 文書をSCDVで学習し、その文書ベクトルをElasticsearchに登録する。
  
    ```
    $ python3 scdv_to_es.py --host [Elasticsearchホスト] --index accident_data --input_csv es_accident_data.csv
    ```

  * pyhthonのmatplotlibで日本語を有効にするための設定
    - 以下にアクセスし、ipaexg00401.zipをダウンロードする。
      https://ipafont.ipa.go.jp/node17
      
    - 解凍する。
    - 解凍したipaexg.ttfファイルをコピーし、matplotlibの設定を変更する。
    
      ```
      $ sudo cp ./ipaexg00401/ipaexg.ttf /usr/local/lib/python3.6/dist-packages/matplotlib/mpl-data/fonts/ttf/
      $ sudo cp ./ipaexg00401/ipaexg.ttf /usr/share/fonts/truetype
      $ sudo vi /usr/local/lib/python3.6/dist-packages/matplotlib/mpl-data/matplotlibrc
      font.family         : IPAexGothic
      
      # フォントのキャッシュを消しておく。
      $ rm -rf ~/.cache/matplotlib/*
      ```

# Jupyter Labで各種デモを実行する。

  * Webブラウザでjupyter labにアクセスする。
    jupyter labが起動していない場合は起動し、コンソールに表示されるURLにアクセスする。
    
    ```
    $ jupyter lab
    ```

  * LDAトピックモデルのサンプル
    LDA_topic_model.ipynbを開く。
    
    - フォントが正しく表示されるか確認するため、いちばん下のセルを実行してみる。
      フォントが正しく表示されない場合、IPAのフォントファイルやmatplitlibのキャッシュなどに問題がある可能性がある。
      
    - LDAトピックモデルの作成セルを実行する
      セル中の「NUM_TOPICS」変数に、分類したいトピック数を入力する。
      
    - LDAトピックモデルのWordCloudセルを実行する
      トピックで分類し、特徴的な単語が表示される。
   
  * GloVeによる類似語検索
    Similarity.ipynbを開く。
    
    - 「GloVeの単語ベクトルによる類似語検索」セルに、類似語を検索したい単語と、抽出したい単語数を設定し、実行する。
      - 変数word : 検索したい単語
      - 変数top_k : 類似する単語を何件抽出するか

  * SCDVによる類似文書抽出
    - Elasticsearch_sim_search.ipynbを開く。
    - Elasticsearchを使用するため、起動しておく。
    - 「キーワード検索」セルに、ホストのアドレスやインデックスの設定を記述する。
    - 「search_word」変数に、検索したい単語を設定する。
       セルを実行すると、この単語を含む文書を検索し、一覧表示する。
    - 「類似検索したい文書を指定する」セルに、上記で検索した文書のうち、
       類似文書を検索したい文書の番号を、変数「target_id」に設定する。ゼロ始まり。
       セルを実行すると、その文書の詳細が表示される。
    - 「類似する文書を検索し、トップ10を表示する」セルを実行する。
       選択した文書と似ている傾向の文書トップ10が表示される。

