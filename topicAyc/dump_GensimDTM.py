"""
Using ldaseqmodel in gensim to establish DTM
"""
# setting up our imports

from gensim.models import ldaseqmodel
from gensim.corpora import Dictionary, bleicorpus
import numpy
from gensim.matutils import hellinger

# loading our corpus and dictionary
try:
    dictionary = Dictionary.load('D:\\Python-TopicModel\\gensim\\docs\\notebooks\\datasets\\news_dictionary')
    # 词典
except FileNotFoundError as e:
    raise ValueError("SKIP: Please download the Corpus/news_dictionary dataset.")
corpus = bleicorpus.BleiCorpus('D:\\Python-TopicModel\\gensim\\docs\\notebooks\\datasets\\news_corpus')
# 词袋
# it's very important that your corpus is saved in order of your time-slices!

time_slice = [438, 430, 456]

ldaseq = ldaseqmodel.LdaSeqModel(corpus=corpus, id2word=dictionary, time_slice=time_slice, num_topics=5)

ldaseq.print_topic_times(topic=0) # evolution of 1st topic

# to check Document - Topic proportions, use `doc-topics`，检测某一篇文档在主题上的概率分布
words = [dictionary[word_id] for word_id, count in corpus[558]]
print (words)
doc = ldaseq.doc_topics(558) # check the 558th document in the corpuses topic distribution
print (doc)

# 检查训练集以外的某个文档在已训练得到的主题上的分布
doc_football_1 = ['economy', 'bank', 'mobile', 'phone', 'markets', 'buy', 'football', 'united', 'giggs']
doc_football_1 = dictionary.doc2bow(doc_football_1)
doc_football_1 = ldaseq[doc_football_1]
print (doc_football_1)

