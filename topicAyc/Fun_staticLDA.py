# encoding = utf-8
"""
LDA modeling functions:
1.dictionary generation
2.coherence,perplexity
"""
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import gensim, pprint
import gensim.corpora as corpora
from gensim.corpora.mmcorpus import MmCorpus
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import os
import numpy as np

import logging
from collections import OrderedDict

from gensim.corpora import TextCorpus, MmCorpus
from gensim import utils, models


# Now estimate the probabilities for the CoherenceModel.
# This performs a single pass over the reference corpus, accumulating
# the necessary statistics for all of the models at once.
class Fun_staticLDA():
    def __init__(self, dpath, nobelow=None, noabove=None):
        self.nobelow = 1 if not nobelow else nobelow
        self.noabove = 1 if not nobelow else noabove
        [docs, dictionary, corpus, mmcorpus] = self.coupusBuilding(dpath)
        self.docs = docs
        self.dictionary = dictionary
        self.corpus = corpus
        self.mmcorpus = mmcorpus


    def coupusBuilding(self, fname):    # 抽取数据，生成字典，语料库
        table = pd.read_table(fname, sep=',', chunksize=1000)
        df_list = []
        for df in table:
            df_list.append(df)
        wdf = pd.concat(df_list, ignore_index=True)

        tdocs = wdf['ptext'].tolist()
        dropind = []    # drop rows with float type
        for i, doc in enumerate(tdocs):
            if isinstance(doc, float):
                dropind.append(i)
        wdf = wdf.drop(dropind)
        wdf = wdf.reset_index()
        docs = wdf['ptext'].tolist()
        docs = [[token for token in doc.split('\n') if len(token) > 1] for doc in docs]

        dictionary = Dictionary(docs)  # id2token:词频-术语; token2id：术语-词频

        # 过滤掉出现次数少于nobelow的，保留在不超过no_above*100%的文档中都出现的词
        dictionary.filter_extremes(no_below=self.nobelow, no_above=self.noabove)

        corpus = [dictionary.doc2bow(doc) for doc in docs]  # unnote for fist time running
        mm_path = 'D:/3policyAyc/_database/_policytxt/corpus_'+str(self.nobelow)+'_'+str(self.noabove)+'.mm'
        if os.path.exists(mm_path):
            mm_corpus = MmCorpus(mm_path)  # load back in to use for LDA training
        else:
            MmCorpus.serialize(mm_path, corpus, id2word=dictionary)  # unnote for first time running
            mm_corpus = MmCorpus(mm_path)  # load back in to use for LDA training
        print('语料库数据初始化完成。')
        return docs, dictionary, corpus, mm_corpus

    def getSamples(self, docs):
        dictionary = Dictionary(docs)
        dictionary.filter_extremes(no_below=self.nobelow, no_above=self.noabove)
        corpus = [dictionary.doc2bow(doc) for doc in docs]
        mm_path = 'D:/3policyAyc/_database/_policytxt/TEMPcorpus_'+str(self.nobelow)+'_'+str(self.noabove)+'.mm'
        MmCorpus.serialize(fname=mm_path, corpus=corpus, id2word=dictionary)
        mm_corpus = MmCorpus(mm_path)
        return mm_corpus, dictionary


    def coherenceRankings(self, cm, coherences):
        avg_coherence = \
            [(num_topics, avg_coherence)
             for num_topics, (_, avg_coherence) in coherences.items()]
        ranked = sorted(avg_coherence, key=lambda tup: tup[1], reverse=True)
        norank = sorted(avg_coherence, key=lambda tup: tup[0], reverse=False)
        print("Ranked by average '%s' coherence:\n" % cm.coherence)
        for item in ranked:
            print("num_topics=%d:\t%.4f" % item)
        print("\nBest: %d" % ranked[0][0])
        return ranked[0][0], list(norank)

    def perplexityRankings(self, models, corpus):
        tls, count = [], 0
        for tn, lda in models.items():
            tls.append((tn, lda.log_perplexity(corpus)))
            count += 1
            print('\r计算perplexity中：{:.2f}%'.format(count * 100 / len(models)), end='')
        ranked = sorted(tls, key=lambda tup: tup[1], reverse=True)

        return ranked[0][0], tls

    def compute_coherence_values(self, limit, start=2, step=3):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        mallet_path = 'C:/Mallet/bin/mallet'
        coherence_values = []
        topicnums = []
        trained_models = OrderedDict()
        for num_topics in range(start, limit, step):
            topicnums.append(num_topics)
            model = gensim.models.wrappers.LdaMallet(mallet_path,
                                                     corpus=self.corpus,
                                                     num_topics=num_topics,
                                                     id2word=self.dictionary,
                                                     workers=4,
                                                     iterations=500)
            trained_models[num_topics] = model
            coherencemodel = CoherenceModel(model=model, texts=self.docs, dictionary=self.dictionary, coherence='u_mass')
            coherence_values.append(coherencemodel.get_coherence())
        df = pd.DataFrame(zip(topicnums,coherence_values))
        df.to_excel('./coherencevary0530_part5.xlsx')
        print('保存一致性取值成功。')
        return trained_models, coherence_values

# # Compute Perplexity for each
# tnlist, perp = [],[]
# for topic_num, model in trained_models.items():
#     tnlist.append(topic_num)
#     perp.append(model.log_perplexity(corpus))
# print(perp)
# minidx = perp.index(min(perp))
# perplexity_best_num = tnlist[minidx]
# best_model2 = trained_models[perplexity_best_num]
# best_model2.save('./_database/_workshop/Jiangsu'+str(perplexity_best_num)+'_perplexitybest.lda', separately=False)
#
#
# # Visualize the topics
# ## for coherence best one
# vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary) ##############################################################for testing
# pyLDAvis.show(vis)
# ## for perplexity best one
# vis = pyLDAvis.gensim.prepare(best_model1, corpus, dictionary)
# pyLDAvis.show(vis)
#
# # U_mass coherence
# top_topics = best_model1.top_topics(corpus) #, num_words=20)
# top_topics = best_model2.top_topics(corpus) #, num_words=20)
#
# pprint(top_topics)
#
# import pandas as pd
#
# f = open('./data/ows-raw.txt',encoding='utf-8')
# reader = pd.read_table(f, sep=',', iterator=True, error_bad_lines=False) #跳过报错行
# loop = True
# chunkSize = 1000
# chunks = []
# while loop:
# 　　try:
# 　　　　chunk = reader.get_chunk(chunkSize)
# 　　　　chunks.append(chunk)
# 　　except StopIteration:
# 　　　　loop = False
# 　　　　print("Iteration is stopped.")
# df = pd.concat(chunks, ignore_index=True)
