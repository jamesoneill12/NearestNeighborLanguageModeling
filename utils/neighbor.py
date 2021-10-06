import numpy as np
from loaders.helpers import load_subword_embeddings
import scipy as sp


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def cosine_sims(A):
    # base similarity matrix (all dot products)
    # replace this with A.dot(A.T).toarray() for sparse representation
    similarity = np.dot(A, A.T)
    # squared magnitude of preference vectors (number of occurrences)
    square_mag = np.diag(similarity)
    # inverse squared magnitude
    inv_square_mag = 1 / square_mag
    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)
    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag
    return cosine


def cos_dist(query_vec, matrix, topn = 10, return_probs = True):
    """
    Compute the cosine distances between each row of matrix and vector.
    """
    v = query_vec.reshape(1, -1)
    # argmin replaced with arg.parition
    cos_distances = sp.spatial.distance.cdist(matrix, v, 'cosine').reshape(-1)
    inds = np.argpartition(cos_distances, topn)
    top_inds = inds[:topn]
    # don't need vecs just the inds
    # most_similar = matrix[top_inds, :]
    probs = softmax(cos_distances[top_inds])
    if return_probs==False:
        return (top_inds)
    return list(top_inds), probs

# this needs to be done after the vocabulary made.
def get_nearest_vocab_neighbor(word2id, id2vec):
    neighbors = {}
    embeddings = np.array(list(id2vec.values()))
    for word, word_id in word2id.items():
        neighbor_ids, similarities = cos_dist(id2vec[word_id], embeddings)
        # kept as list because np array changes ids to float
        neighbors[word_id] = [neighbor_ids, similarities]
    return neighbors

# OLD
def get_nearest_neighbor(word, embs):
    # I also want sample probabilities so perhaps best to softmax the similarities
    similarities = []; neighbors = []; cnt_words = 0
    for neighbor, similarity in embs.wv.most_similar(word, topn = 10):
        similarities.append(similarity)
        neighbors.append(neighbor)
    neighbor_info = [np.vstack(neighbors), softmax(np.array(similarities))]
    print("{} \% of word neighbors retrieved".format(cnt_words * 100/float(len(embs))))
    return neighbor_info


def get_nearest_neighbors(word2vec):
    embs = load_subword_embeddings()
    # I also want sample probabilities so perhaps best to softmax the similarities
    neighbor_lookup = {}
    cnt_words = 0
    for (word, vec) in word2vec.items():
        similarities = []
        neighbors = []
        if word in embs:
            cnt_words+=1
            for neighbor, similarity in embs.wv.most_similar(word, topn = 10):
                similarities.append(similarity)
                neighbors.append(neighbor)
            # zip removed because we do not want list of tuples now
            neighbor_lookup[word] = [neighbors, softmax(np.array(similarities))]
    print("{} \% of word neighbors retrieved".format(cnt_words * 100/float(len(word2vec))))
    return neighbor_lookup
