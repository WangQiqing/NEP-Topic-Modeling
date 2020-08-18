# python3 -m spacy download en

import numpy as np
import pandas as pd
import os, re, nltk, spacy, gensim
from topicAyc.Fun_DTM import Fun_DTM
from tqdm import tqdm
from multiprocessing import Pool

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

class dtmAyc():
    @staticmethod
    def save_significance(rpath, fpathlist):
        for i in tqdm(range(0, len(fpathlist))):
            dynamicLDA = Fun_DTM(os.path.join(rpath,fpathlist[i]))
            dtmmod = gensim.models.wrappers.DtmModel.load(  # get dynamic topic model
                'D:\\3policyAyc\\_database\\_workshop\\dtmprovs_' + str(i + 1) + '29topics.model')

            df = dynamicLDA.getTopicsignificance(dtmmod)
            df.to_excel('D:\\2020止水禪心\\0.3新能源政策挖掘\\论文\\图表\\'+fpathlist[i].strip('.csv').strip('Wordlist_')+
                        '_topicsignificance1.xlsx')


    @staticmethod
    def save_topics(fpathlist):
        writer = pd.ExcelWriter('D:\\2020止水禪心\\0.3新能源政策挖掘\\论文\\图表\\_topicTerms.xlsx')
        topn = 15
        for i in tqdm(range(len(fpathlist))):
            dtmmod = gensim.models.wrappers.DtmModel.load(  # get dynamic topic model
                'D:\\3policyAyc\\_database\\_workshop\\newdtmprovs_' + str(i + 1) + '29topics.model')
            topdict = {}
            for topicid in range(29):
                topics_over_time = dtmmod.show_topic(topicid=topicid, topn=topn, time=0)
                terms = [tup[1] for tup in topics_over_time]
                topdict.update({'Topic' + str(topicid + 1): terms})
            tdf = pd.DataFrame(topdict)
            tdf.to_excel(writer, sheet_name='Sheet_provs'+str(i+1))
            writer.save()

    @staticmethod
    def term_variations(rpath, fpathlist, tls, no):
        provdic = {1:'Inner Mongolia', 2:'Sichuan', 3:'Nationwide',4:'Shandong',5:'Guangdong', 6:'Sinkiang', 7:'Jiangsu'}
        dfls = []
        for i in tqdm(range(len(fpathlist))):
            # terms = ['新能源','风电','太阳能','低碳','减排']
            terms=tls
            termvaries = {}
            dynamicLDA = Fun_DTM(os.path.join(rpath,fpathlist[i]))
            dtmmod = gensim.models.wrappers.DtmModel.load(  # get dynamic topic model
                'D:\\3policyAyc\\_database\\_workshop\\dtmprovs_' + str(i + 1) + '29topics.model')
            doc_topic, topic_term, doc_lengths, term_frequency, vocab = \
                dtmmod.dtm_vis(dynamicLDA.dcorpus,time=list(range(10)))# model results
              # save to one single excel
            for term in terms:
                try:
                    term_ind = vocab.index(term)    # locate topic for corresponding term
                except ValueError:
                    print('The given term {} is invalid.'.format(term))
                    continue
                termfreqs = topic_term[:,term_ind,:]    # get term frequency
                fit_topicid = np.argmax(np.average(termfreqs,1))    # locate most significant topic
                termvaries.update({term:termfreqs[fit_topicid].tolist()})
            termvaries.update({'Provs': [provdic[i+1]]*10})
            termvaries.update({'Years': list(range(2010,2020))})
            termvarydf = pd.DataFrame(termvaries)
            dfls.append(termvarydf)
        dfall = pd.concat(dfls, ignore_index=True)
        savpath = 'D:\\2020止水禪心\\0.3新能源政策挖掘\\论文\\图表\\term_variations_part' + str(no) + '.xlsx'
        dfall.to_excel(savpath, sheet_name=provdic[i+1])


if __name__ == '__main__':
    path = 'D:/3policyAyc/_database/_policytxt'
    fpathlist = ['Wordlist_内蒙古5.csv', 'Wordlist_四川5.csv', 'Wordlist_国家5.csv',
                 'Wordlist_山东5.csv', 'Wordlist_广东5.csv', 'Wordlist_新疆5.csv', 'Wordlist_江苏5.csv']
    # dyDTM = Fun_DTM(os.path.join(path, fpathlist[6]))
    # dtm = gensim.models.wrappers.DtmModel.load('D:\\3policyAyc\\_database\\_workshop\\dtmprovs_729topics.model')
    # for i in tqdm(range(len(fpathlist))):
    #     dynamicLDA = Fun_DTM(os.path.join(path,fpathlist[i]))
    #     model = dynamicLDA.getDTM(fpathlist[i].strip('5.csv').strip('Wordlist_')+'.model')
    termls = [["节能", "减排", "碳排放", "风电", "生物质", "新能源", "新兴产业", "煤炭", "新能源汽车", "环保"],
    ["节能", "低碳", "碳排放", "风能", "太阳能", "生物质能", "可再生能源", "装备", "煤炭", "电动汽车", "环保"],
    ["节能", "低碳", "风能", "太阳能", "生物质能", "新能源", "新兴产业", "新能源汽车", "环保"],
    ["节能", "减排", "碳排放", "风电", "生物质", "新能源", "新兴产业", "煤炭", "新能源汽车", "环保", "低碳", "风能", "太阳能", "生物质能", "可再生能源",
     "装备", "电动汽车"]]
    for i,ter in enumerate(termls):
        dtmAyc.term_variations(path, fpathlist, ter, i+1)
    # dtmAyc.save_significance(path, fpathlist)
    # dtmAyc.save_topics(fpathlist)

    # train dtm models from 5 to 29 sep 2
    # provs = ['广东','新疆','江苏']
    # for provn in provs:
    #     pool = Pool(4)
    #     modelname = [provn + str(topicnum) + '_dtm.model' for topicnum in range(5, 30, 2)]
    #     dyDTM = pool.map(Fun_DTM, modelname)


    # for i in range(10):
    #     print('时间={}时的主题：'.format(i))`
    #     for nums in range(29):
    #         print(dtm.show_topic(topicid=nums,time=i, topn=5))




# from gensim.models.coherencemodel import CoherenceModel
# from topicAyc.Fun_DTM import Fun_DTM
# import os
# path = 'D:/3policyAyc/_database/_policytxt'
# dynamicLDA = Fun_DTM(os.path.join(path, 'Wordlist_内蒙古5.csv'))
# dtmmod = gensim.models.wrappers.DtmModel.load('D:\\3policyAyc\\_database\\_workshop\\dtmprovs_129topics.model')
# cm = CoherenceModel(model=dtmmod, corpus=dynamicLDA.dcorpus, coherence='u_mass')
# coherence = cm.get_coherence()  # get coherence value
#
#
# from topicAyc.Fun_staticLDA import Fun_staticLDA
# c = Fun_staticLDA.compute_Cumass(1,2)
# import pyLDAvis
# from importlib import reload
#
# doc_topic, topic_term, doc_lengths, term_frequency, vocab = model.dtm_vis(time=0, corpus=corpus)
# vis_wrapper = pyLDAvis.prepare(topic_term_dists=topic_term, doc_topic_dists=doc_topic, doc_lengths=doc_lengths, vocab=vocab, term_frequency=term_frequency)
# pyLDAvis.display(vis_wrapper)
# pyLDAvis.show(vis_wrapper)
#
# doc_topic, topic_term, doc_lengths, term_frequency, vocab = model.dtm_vis(time=1, corpus=corpus)
# vis_wrapper = pyLDAvis.prepare(topic_term_dists=topic_term, doc_topic_dists=doc_topic, doc_lengths=doc_lengths, vocab=vocab, term_frequency=term_frequency)
# pyLDAvis.display(vis_wrapper)
# pyLDAvis.show(vis_wrapper)
