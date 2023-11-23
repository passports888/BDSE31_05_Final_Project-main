import pandas as pd
import scipy.sparse
from sklearn.neighbors import NearestNeighbors


def knn_model(i):
    sparse_matrix = scipy.sparse.load_npz('sparse_matrix.npz')
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    encode_articles = pd.read_csv('encode_articles.csv')
    encode_articles = encode_articles.article_id.to_list()
    model_knn.fit(sparse_matrix)
    
    i = encode_articles.index(i)
    x = sparse_matrix[i,:].toarray().reshape(1,-1)
    CF = model_knn.kneighbors(x, 6,return_distance=False)
    
    Recommender_item = []
    for ii in CF[0]:
        if ii != i:
            item=encode_articles[ii]
            Recommender_item.append(item)
    
    return Recommender_item