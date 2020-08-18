from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
import gensim
import pandas as pd
from gensim.parsing.preprocessing import preprocess_documents
from multiprocessing import Pool
from functools import partial
import math
import numpy as np
from topicAyc.Fun_staticLDA import Fun_staticLDA

# implements the UMass coherence in Mimno et al. 2011 - Optimizing Semantic Coherence in Topic Models
def cooccur_df_ws(w1, w2, corpus_dense, w2ids):
    """
    Returns the co-document frequency of two words
    """
    w1_id, w2_id = w2ids.token2id.get(w1), w2ids.token2id.get(w2)
    co_freq_array = (corpus_dense[:, [w1_id, w2_id]] > 0).sum(axis=1).A1
    return np.count_nonzero(co_freq_array == 2)


def word_lst_coherence(corpus_dense, w2ids, word_list):
    """
    Given a sequence of words, calculate the overall UMASS-coherence (eq 1 in the paper)
    """
    C = 0
    for i, w_rare in enumerate(word_list[1:]):
        for j, w_common in enumerate(word_list):
            # m = rare word, l = common word in the Mimno paper
            if i >= j:  # make sure the index of w_common is ahead of w_rare
                D_m_l = cooccur_df_ws(w_rare, w_common, corpus_dense, w2ids)
                D_l = w2ids.dfs[w2ids.token2id.get(w_common)]
                C = C + math.log((D_m_l + 1) / D_l)
    return C


def topic_model_coherence(topicwords, corpus_dense, w2ids):
    """
    Calculate the average coherence of all the topics in a fitted LDA model
    """
    topic_coherences = []
    topic_coherences.append(word_lst_coherence(corpus_dense=corpus_dense,
                                              w2ids=w2ids,
                                              word_list=topicwords))

    return np.mean(topic_coherences)


if __name__ == '__main__':
    # use the newsgroup data as corpus
    path = 'D:/3policyAyc/_database/_policytxt/Wordlist_内蒙古5.csv'
    statLDA = Fun_staticLDA(path)
    corpus = statLDA.corpus
    dictionary = statLDA.dictionary
    lda = gensim.models.LdaModel(corpus, num_topics=20, id2word=dictionary)
    # convert gensim corpus to a sparse document-term matrix for coherence measure
    corpus_dense = gensim.matutils.corpus2csc(corpus, num_terms=len(dictionary.keys()))
    corpus_dense = corpus_dense.astype(int)
    corpus_dense = corpus_dense.transpose()
    topicwordls = []  # a list of list of top words under each topic
    for i in range(lda.num_topics):
        topicwordls.append(
            [x[0] for x in lda.show_topic(i, topn=20)]
        )

    pool = Pool(4)
    vals = pool.map(
        partial(topic_model_coherence, corpus_dense=corpus_dense, w2ids=dictionary),
                topicwordls,
    )
    pool.close()
    pool.join()
    coheval = np.mean(vals)
    print(coheval)
    # a = topic_model_coherence(model=lda, corpus_dense=corpus_dense, w2ids=dictionary, n_top_words=300)