import elasticsearch
import json, argparse, scdv

class elasticsearchClient():
    def __init__(self, host, port, index):
        self.host = host
        self.port = port
        self.index = index
        self.client = elasticsearch.Elasticsearch(self.host + ":" + self.port)

    def update(self, row_id, body):
        response = self.client.update(
                index = self.index, 
                id = row_id, 
                body = body)
        print(response)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str)
    parser.add_argument('--port', type=str, default='9200')
    parser.add_argument('--index', type=str)
    parser.add_argument('--input_csv', type=str)

    return parser.parse_args()

def main(args):
    client = elasticsearchClient(args.host, args.port, args.index)

    scdv_vec = scdv.build_model(args.input_csv, 20, "gmm_cluster.pkl", "gmm_prob_cluster.pkl")

    for row_id, vector in scdv_vec:
        client.update(row_id, {'doc':{'scdv_vector':vector.tolist()}})

if __name__ == '__main__':
    main(parse_args())
