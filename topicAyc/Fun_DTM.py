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
from gensim.models.wrappers.dtmmodel import DtmModel
import os
import numpy as np
import re


class Fun_DTM():
    def __init__(self, fpath='', num_topics=None, nobelow=None, noabove=None):
        # path = 'D:/3policyAyc/_database/_policytxt'
        # fpathlist = ['Wordlist_内蒙古5.csv', 'Wordlist_四川5.csv', 'Wordlist_山东5.csv',
        #              'Wordlist_广东5.csv', 'Wordlist_新疆5.csv', 'Wordlist_江苏5.csv']
        # dgls = {'内蒙古':1, '四川':2, '山东':4, '广东':5, '新疆':6, '江苏':7}
        # for i,name in enumerate(dgls.keys()):
        #     if name in fpath:
        #         datapath = os.path.join(path, fpathlist[i])
        # if not os.path.exists(datapath):
        #     print('路径不存在。')
        #     return
        self.nobelow = 15 if nobelow is None else nobelow
        self.noabove = 0.7 if noabove is None else noabove
        self.topicnums = 29 if num_topics is None else num_topics
        [ddocs, ddictionary, dcorpus, dsubcorpora, dsubdictionaries, dtimeslices] = self.coupusBuilding(fpath)
        self.ddocs = ddocs
        self.ddictionary = ddictionary
        self.dcorpus = dcorpus
        self.dtimeslices = dtimeslices
        self.dsubcorpora = dsubcorpora
        self.dsubdictionaries = dsubdictionaries
        self.topicsig_overtime = np.zeros((self.topicnums, 10)) # topic significance of each year
        # self.dtmModel = self.getDTM(fpath)

    def coupusBuilding(self, fname):  # 抽取数据，生成字典，语料库
        # path = 'D:/3policyAyc/_database/_policytxt'
        # fpathlist = ['Wordlist_内蒙古5.csv', 'Wordlist_四川5.csv','Wordlist_国家5.csv',
        #      'Wordlist_山东5.csv','Wordlist_广东5.csv','Wordlist_新疆5.csv','Wordlist_江苏5.csv']
        # fname = os.path.join(path, fpathlist[0])
        def getcorpus(docs):
            tdic = Dictionary(docs)
            tdic.filter_extremes(no_below=self.nobelow,no_above=self.noabove)
            return [tdic.doc2bow(doc) for doc in docs], tdic
        df = pd.read_csv(fname)
        timeslices, ptexts, subcorpora, subdictionaries = [],[],[],[]
        alltexts = []
        for t in np.arange(2010, 2020):
            tempdf = df.loc[(df["year"] == t)]
            temptext = list(filter(lambda x: not isinstance(x, float),
                                   tempdf['ptext'].tolist()))
            temptextls = [[token for token in doc.split('\n') if len(token) > 1]
                                      for doc in temptext]
            subcorpus, subdictionary = getcorpus(temptextls)
            subcorpora.append(subcorpus)
            subdictionaries.append(subdictionary)
            timeslices.append(len(temptext))
            alltexts.extend(temptextls)

        corpus, dictionary = getcorpus(alltexts)
        print('语料库数据初始化完成。')
        return alltexts, dictionary, corpus, subcorpora, subdictionaries, timeslices

    # def getDTM(self, fname=''):
    #     modelpath = os.path.join('D:/3policyAyc/_database/_workshop/',fname)
    #     matchtopn = re.search(r'\d+', fname)
    #     topnum = int(matchtopn.group())
    #     provn = re.search(r'[\u4e00-\u9fa5]+', fname).group()
    #     if not os.path.exists(modelpath):
    #         print('开始训练{}的{}主题dtm模型...'.format(provn, topnum))
    #         model = DtmModel(r"D:\dtm-win64.exe",
    #                          self.dcorpus, self.dtimeslices,
    #                          num_topics=topnum,
    #                          id2word=self.ddictionary)
    #         model.save(modelpath)
    #         print('{}的模型训练成功。'.format(provn))
    #         return model
    #     else:
    #         try:
    #             dtmod = DtmModel.load(modelpath)
    #             return dtmod
    #         except FileNotFoundError:
    #             return


    def getTopicsignificance(self, model):
        """
        Get topic significance values from a cluster of texts
        :return: time-topics matrix（10*29）
        """
        if not type(model) is DtmModel:
            print('非DtmModel类')
            return
        sig_matrix, nomalized = [],[]
        doc_topic,_,_,_,_ = model.dtm_vis(self.dcorpus, list(range(10)))    # 10*V
        for i in range(1, len(self.dtimeslices)+1):
            pre = sum(self.dtimeslices[:i-1])
            pos = sum(self.dtimeslices[:i])
            sub_dt = doc_topic[pre:pos,:].sum(0)
            sub_dt = sub_dt/sum(sub_dt)
            sig_matrix.append(sub_dt)
        topicls = ['Topic' + str(num + 1) for num in range(29)]
        df = pd.DataFrame(sig_matrix, columns=topicls)
        # max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
        # df = df.apply(max_min_scaler, axis=0)  # 纵向归一化，对时间序列上每个主题显著性序列归一化
        return df