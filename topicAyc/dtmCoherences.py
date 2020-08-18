"""
calculate coherence values of dtm, using Umass method
"""
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
import gensim
import pandas as pd
from gensim.parsing.preprocessing import preprocess_documents
from multiprocessing import Pool
from functools import partial
import math
import numpy as np
from topicAyc.Fun_DTM import Fun_DTM
from tqdm import tqdm
from collections import OrderedDict


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

    return np.mean(topic_coherences)    # get average coherence of each topn word over its topic


if __name__ == '__main__':
    modelname = ['山东' + str(topicnum) + '_dtm.model' for topicnum in range(5, 30, 2)]
    # models = OrderedDict()
    for mpa in modelname:
        dynamicLDA = Fun_DTM(mpa)
        dtm = dynamicLDA.dtmModel
     # path = 'D:/3policyAyc/_database/_policytxt/Wordlist_内蒙古5.csv'
    # name =
    # for i in range(5,29,2):
    #     mpath = 'D:/3policyAyc/_database/_workshop/dtmprovs_329topics.model'
    # for i, fname in mode
    # dynamicLDA = Fun_DTM(path)
    # print('开始训练dtm...')
    # model = gensim.models.wrappers.DtmModel(r"D:\dtm-win64.exe",
    #                  dynamicLDA.dcorpus, dynamicLDA.dtimeslices,
    #                  num_topics=dynamicLDA.topicnums,
    #                  id2word=dynamicLDA.ddictionary)
    # model.save(mpath)
    #     dtm = gensim.models.wrappers.DtmModel.load(mpath)

        corpus = dynamicLDA.dcorpus
        dictionary = dynamicLDA.ddictionary
        corpus_dense = gensim.matutils.corpus2csc(corpus,
                                                  num_terms=len(dictionary.keys()))     # doc-term matrix
        corpus_dense = corpus_dense.astype(int)
        corpus_dense = corpus_dense.transpose()

        coherence_over_time = []
        for i in tqdm(range(10)): # for time 2010 to 2019
            topicwordls = []  # a list of list of top words under each topic of each time
            for j in range(dtm.num_topics):
                topicwordls.append(
                    [x[1] for x in dtm.show_topic(topicid=j, time=i, topn=300)]    # timeslice = 0
                )
            # b = topic_model_coherence(topicwordls[0], corpus_dense=corpus_dense, w2ids=dictionary)
            pool = Pool(4)
            vals = pool.map(
                partial(topic_model_coherence, corpus_dense=corpus_dense, w2ids=dictionary),
                        topicwordls,
            )   # get topic coherence(by average C(w, Tw)) list
            pool.close()
            pool.join()
            coheval = np.mean(vals)
            coherence_over_time.append(coheval)
        print('{}个主题模型的一致性值为：{}'.format(dtm.num_topics, coherence_over_time))