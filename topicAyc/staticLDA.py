from gensim import models
from topicAyc.Fun_staticLDA import Fun_staticLDA

path = 'D:/3policyAyc/_database/_policytxt/Wordlist_all6.csv'
staticLDA = Fun_staticLDA(dpath=path, nobelow=20, noabove=0.8)

#MalletLDA
mallet_path = 'C:/Mallet/bin/mallet'
model_mallet = models.wrappers.LdaMallet(mallet_path,
                                     corpus=staticLDA.mmcorpus,
                                     num_topics=29,
                                     id2word=staticLDA.dictionary,
                                     workers=4,
                                     iterations=500)
model_mallet.save('D:/3policyAyc/_database/_interresults/model_mallet_allcorpus_29_0.8.lda')

#MulticoreLDA

model_multicore = models.LdaMulticore(
    staticLDA.corpus, id2word=staticLDA.dictionary, num_topics=29, workers=4,
    passes=10, iterations=500, random_state=42, eval_every=None,
    alpha='asymmetric',  # shown to be better than symmetric in most cases
    decay=0.5, offset=64  # best params from Hoffman paper
)
model_multicore.save('D:/3policyAyc/_database/_interresults/model_multicore_allcorpus_29_0.8.lda')


model_multicore = models.ldamulticore.LdaMulticore.load('D:/3policyAyc/_database/_interresults/model_multicore_allcorpus_29_0.8.lda')
model_mallet = models.wrappers.LdaMallet.load('D:/3policyAyc/_database/_interresults/model_mallet_allcorpus_29_0.8.lda')

# show topics
a = model_mallet.print_topics(num_topics=29, num_words=10)
b = model_multicore.print_topics(num_topics=29, num_words=10)

model_multicore.show_topics(num_topics=29, num_words=10, log=False, formatted=False)
model_mallet.show_topics(num_topics=5, num_words=10, log=False, formatted=False)

# save topics to excel
import os,re
from gensim import models
import pandas as pd
path = 'D:/3policyAyc/_database/_interresults'
filenames = ['model_multicore_allcorpus_29_0.8.lda','model_multicore_allcorpus_29.lda',
             'model_multicore_allcorpus_29_0.75.lda', 'model_multicore_allcorpus_29_0.65.lda',
              'model_mallet_allcorpus_29_0.8.lda', 'model_mallet_allcorpus_29.lda',
              'model_mallet_allcorpus_29_0.75.lda', 'model_mallet_allcorpus_29_0.65.lda']

for fn in filenames:
    if 'multicore' in fn:
        model = models.LdaMulticore.load(os.path.join(path, fn))
    else:
        model = models.wrappers.LdaMallet.load(os.path.join(path, fn))
    # writer = pd.ExcelWriter('D:/3policyAyc/_database/_interresults/staticMulticoreLDA_all.xlsx')
    topn = 20
    topdict = {}
    for topic in range(29):
        temp = model.show_topic(topic, topn)
        terms = [tup[0] for tup in temp]
        topdict.update({'Topic' + str(topic + 1): terms})
    name_str = fn.strip('.lda')
    write_path = os.path.join(path, 'staticLDA_all_'+name_str+'.xlsx')
    tdf = pd.DataFrame(topdict)
    tdf.to_excel(write_path)

################################################################################################################
# for provincial and central policy analysis
import os, gensim
import pandas as pd
from gensim import models

def malletTopic(fpath):
    wdf = pd.read_csv(fpath)
    docs = wdf['ptext'].tolist()
    docs = [[token for token in doc.split('\n') if len(token) > 1]
            for doc in docs if not isinstance(doc, float)]
    dictionary = gensim.corpora.Dictionary(docs)  # id2token:词频-术语; token2id：术语-词频
    # 过滤掉出现次数少于nobelow的，保留在不超过no_above*100%的文档中都出现的词
    dictionary.filter_extremes(no_below=20, no_above=0.7)
    corpus = [dictionary.doc2bow(doc) for doc in docs]  # unnote for fist time running
    mm_path = './tempcorpus.mm'
    gensim.corpora.MmCorpus.serialize(mm_path, corpus, id2word=dictionary)  # unnote for first time running
    mm_corpus = gensim.corpora.MmCorpus(mm_path)  # load back in to use for LDA training
    mallet_path = 'C:/Mallet/bin/mallet'
    model_mallet = models.wrappers.LdaMallet(mallet_path,
                                             corpus=mm_corpus,
                                             num_topics=29,
                                             id2word=dictionary,
                                             workers=4,
                                             iterations=500)
    try:
        savepath = 'D:/3policyAyc/_database/_interresults/staticLDA_mallet_' +\
                   fpath.split('Wordlist_')[1].strip('.csv') + '.lda'
        model_mallet.save(savepath)
        print('保存模型成功。')
    except:
        print('保存模型失败。')
    return model_mallet


def multicoreTopic(fpath):
    wdf = pd.read_csv(fpath)
    docs = wdf['ptext'].tolist()
    docs = [[token for token in doc.split('\n') if len(token) > 1]
            for doc in docs if not isinstance(doc, float)]
    dictionary = gensim.corpora.Dictionary(docs)  # id2token:词频-术语; token2id：术语-词频
    # 过滤掉出现次数少于nobelow的，保留在不超过no_above*100%的文档中都出现的词
    dictionary.filter_extremes(no_below=20, no_above=0.7)
    corpus = [dictionary.doc2bow(doc) for doc in docs]  # unnote for fist time running
    model_multicore = models.LdaMulticore(corpus, id2word=dictionary, num_topics=29, workers=4,
        passes=10, iterations=500, random_state=42, eval_every=None,
        alpha='asymmetric',  # shown to be better than symmetric in most cases
        decay=0.5, offset=64  # best params from Hoffman paper
    )
    try:
        savepath = 'D:\\3policyAyc\\_interresults\\staticLDA_multicore_' + fpath.split('_')[1].strip('.csv') + '.lda'
        model_multicore.save(savepath)
        print('保存模型成功。')
    except:
        print('保存模型失败。')
    return model_multicore


def save2excel(spath, model):
    topn = 20
    topdict = {}
    for topic in range(29):
        temp = model.show_topic(topic, topn)
        terms = [tup[0] for tup in temp]
        topdict.update({'Topic' + str(topic + 1): terms})
    tdf = pd.DataFrame(topdict)
    tdf.to_excel(spath)
    print('保存主题成功。')


path = 'D:/3policyAyc/_database/_policytxt'
fpathlist = ['Wordlist_内蒙古5.csv', 'Wordlist_四川5.csv','Wordlist_国家5.csv',
             'Wordlist_山东5.csv','Wordlist_广东5.csv','Wordlist_新疆5.csv','Wordlist_江苏5.csv']

for fp in fpathlist:
    fn = os.path.join(path, fp)
    if os.path.exists(fn):
        model = malletTopic(fn)
        name_str = fp.split('_')[1].strip('.csv')
        savpath = 'D:/3policyAyc/_database/_interresults/STATICLDA'+name_str+'.xlsx'
        save2excel(savpath,model)
    else:
        print(fn+'不存在！！')


# for test
path = r'D:\3policyAyc\Visualization\topicNetwork\mallet29.lda'
import gensim
mod = gensim.models.wrappers.LdaMallet.load(path)
mod.print_topics(num_topics=10,num_words=5)
ten_topics = mod.show_topics(num_topics=28, num_words=30000,formatted=False)
sum_termprob = []
for numtop, wordtup in ten_topics:
    tempsum= sum([prob[1] for prob in wordtup])
    temptup = (numtop, tempsum)
    sum_termprob.append(temptup)

from pprint import pprint
pprint(mod.print_topics(num_topics=28, num_words=1))
pprint(sum_termprob)