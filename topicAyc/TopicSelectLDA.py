# encoding = utf-8
"""
Determination of LDA topic numbers
"""
from __future__ import print_function
import pandas as pd
from gensim.corpora import Dictionary
import os
import gensim
import random

import logging
from collections import OrderedDict

from gensim.corpora import TextCorpus, MmCorpus
from gensim import utils, models
from topicAyc.Fun_staticLDA import Fun_staticLDA

logging.basicConfig(level=logging.ERROR)  # disable warning logging


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Some useful utility functions in case you want to save your models.

models_dir = r"D:\3policyAyc\_database\_workshop"
def save_models(named_models):
    for num_topics, model in named_models.items():
        model_path = os.path.join(models_dir, 'lda-MALLET-k%d.lda' % num_topics)
        model.save(model_path, separately=False)


def load_models():
    trained_models = OrderedDict()
    for num_topics in range(5, 31, 5):
        model_path = os.path.join(models_dir, 'lda-all7-k%d.lda' % num_topics)
        print("Loading LDA(k=%d) from %s" % (num_topics, model_path))
        trained_models[num_topics] = models.LdaMulticore.load(model_path)

    return trained_models
from gensim import models, utils

# load LDA model(class:LDAmodel)
modelpath = 'D:\\3policyAyc\\_database\\_workshop\\lda-all0-k10.lda'

#models.wrappers.LdaMallet.load(modelpath)

#models.LdaModel.load(modelpath)


# stopwords = []
# fp = open('D:/3policyAyc/_database/_auxdata/rmvs_for_wordcloud.txt', 'r', encoding='utf-8')
# for line in fp.readlines():
#     if line != '' and line != '\n':
#         stopwords.append(line.strip('\n'))
# stopwords = list(set(stopwords))
# noabov = [0.6, 0.5, 0.4, 0.3, 0.25]
# for i in range(5):


datapath = 'D:/3policyAyc/_database/_policytxt/Wordlist_all5.csv'
staticLDA = Fun_staticLDA(datapath, nobelow=5, noabove=0.7)  # initialize static LDA model

"""Train LDA model"""
# training,holding, and testing samples
rdocs = staticLDA.docs
traindocs = rdocs[:15000]
testdocs = rdocs[15000:]
training_corpus, training_dictionary = staticLDA.getSamples(traindocs)
testing_corpus, testing_dictionary = staticLDA.getSamples(testdocs)
# Start training using multicore
trained_models = OrderedDict()

for num_topics in range(26,37,1):
    print("Training LDA(k=%d)" % num_topics)
    lda = models.LdaMulticore(
        training_corpus, id2word=training_dictionary, num_topics=num_topics, workers=4,
        passes=10, iterations=100, random_state=42, eval_every=None,
        alpha='asymmetric',  # shown to be better than symmetric in most cases
        decay=0.5, offset=64  # best params from Hoffman paper
    )
    trained_models[num_topics] = lda

# writer = pd.ExcelWriter('D:/3policyAyc/_database/_interresults/0525LDAtest.xlsx')
# topn = 50
# for num_topic,ldamod in trained_models.items():
#     topdict = {}
#     for topic in range(num_topic):
#         temp = ldamod.show_topic(topic, topn)
#         terms = [tup[0] for tup in temp]
#         topdict.update({'Topic' + str(topic + 1): terms})
#     tdf = pd.DataFrame(topdict)
#     tdf.to_excel(writer,sheet_name='Topic'+str(num_topic))
# writer.save()

save_models(trained_models)

trained_models = load_models()  # for testing

# compare different topic numbers from coherence and perplexity
# coherence calculation
cm = models.CoherenceModel.for_models(
    trained_models.values(), staticLDA.dictionary, texts=staticLDA.docs, coherence='c_v')

coherence_estimates = cm.compare_models(trained_models.values())
coherences = dict(zip(trained_models.keys(), coherence_estimates))

[cohrence_best_num, cohtuples] = staticLDA.coherenceRankings(cm, coherences)  # rank by average coherence
best_model1 = trained_models[cohrence_best_num]
cohdf = pd.DataFrame(cohtuples, columns=['topic_numbers','cohrence_values'])    # save coherence values
best_model1.save('./_database/_workshop/Static'+str(cohrence_best_num)+'_10to50cohebest_FINAL.lda', separately=False)

# perplexity calculation

tls,count = [],0
for tn, lda in trained_models.items():
    tls.append((tn, lda.log_perplexity(testing_corpus)))
    count += 1
    print('\r计算perplexity中：{:.2f}%'.format(count*100/len(trained_models)), end='')
ranked = sorted(tls, key=lambda tup: tup[1], reverse=True)

[perplexity_best_num, perptuples] = staticLDA.perplexityRankings(trained_models)  # rank by log perplexity
perplexity_best_num = ranked[0][0]
best_model2 = trained_models[perplexity_best_num]
perpdf = pd.DataFrame(ranked, columns=['topic_numbers', 'perp_values'])
perpdf.to_excel('./_database/_interresults/迷惑度-final1.xlsx')
best_model2.save('./_database/_workshop/Static'+str(perplexity_best_num)+'_10to50perpbest_final.lda', separately=False)

combdf = pd.merge(cohdf, perpdf, on='topic_numbers')
combdf.to_excel('./_database/_interresults/测试一致性和迷惑度-final.xlsx')



# topic calculation，Output results to excel
model = models.LdaMulticore.load('D:/3policyAyc/_database/_workshop/Static24_10to50cohebest.lda')
writer = pd.ExcelWriter('D:/3policyAyc/_database/_interresults/0525LDAresults.xlsx')
topn = 50
for num_topic,ldamod in trained_models.items():
    topdict = {}
    for topic in range(24):
        temp = model.show_topic(topic, topn)
        terms = [tup[0] for tup in temp]
        topdict.update({'Topic' + str(topic + 1): terms})
    tdf = pd.DataFrame(topdict)
    tdf.to_excel(writer)
    writer.save()
    # tdf.to_excel(writer,sheet_name='Topic'+str(num_topic))
# writer.save()





