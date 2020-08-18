"""preprocess data for Wordcloud"""

import unicodedata, re, os
from dataPrep.Func_extraction import Func_extraction
from dataPrep.Func_chunking import Func_chunking
import pandas as pd
from timeit import default_timer as timer


"""start extraction"""
extractor = Func_extraction()
chunker = Func_chunking()

'''标题识别'''

provnames = ['内蒙古', '四川', '国家','山东','广东','新疆','江苏','四川']
for provname in provnames:
    path = 'D:/3policyAyc/_database/_policytxt/Raw_'+provname+'1.csv'
    print('开始处理{}的政策文本,文本路径：{}'.format(provname, path)+'>>>>'*7)
    df = pd.read_csv(path)
    titles = df['title'].tolist()

    stopdocs = []
    fp = open('D:/3policyAyc/_database/_auxdata/rmvs_Docs.txt', 'r', encoding='utf-8')
    for line in fp.readlines():
        stopdocs.append(line.strip('\n'))
    fp.close()

    # 带有停用词的标题
    specialtits, indlist = [],[]
    for w in set(stopdocs):
        for tit in titles:
            if w in tit:
                # print(tit)
                specialtits.append(tit)
                indlist.append(titles.index(tit))

    # 删除含停用标题的数据
    raw_df = df.drop(index = indlist, axis=1)
    raw_df = raw_df.reset_index(drop=True)
    del specialtits, indlist

    """start preprocessing,word segmentation, POS tagging"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    print('开始执行去除停用词和分词操作,提取所有名动词、名词、形容词和专有名词'+'>>>>'*7)
    tic1 = timer()
    count = 0
    ptlist = raw_df['ptext'].tolist()
    doclen = len(ptlist)
    newptlist = []
    for doc in ptlist:
        newptlist.append('\n'.join(chunker.cutwithPOS(doc, False)))
        count += 1
        print('\r提取进度：{:.2f}%'.format(count * 100 / doclen), end='')
    toc1 = timer()
    print('分词抽取完毕！用时'+str(toc1-tic1)+'秒！')

    raw_df['ptext'] = newptlist
    # df.to_excel(r'D:\3policyAyc\_database\_policytxt\Wordlist_'+list(provnames.keys())[namenum-1]+'.xlsx')
    raw_df.to_csv(r'D:\3policyAyc\_database\_policytxt\Wordlist_'+provname+'_forWordCloud.csv', encoding="utf_8_sig", index=False)

# 合并数据
import pandas as pd
provnames = ['内蒙古', '四川', '国家','山东','广东','新疆','江苏']
dfls0 = []
for prov in provnames:
    tempdf = pd.read_csv(r'D:\3policyAyc\_database\_policytxt\Wordlist_'+prov+'_forWordCloud.csv')
    dfls0.append(tempdf)
# dfls0[1].drop(index=)
dfall0 = pd.concat(dfls0, ignore_index=True)
dfall0.to_csv(r'D:\3policyAyc\_database\_policytxt\Wordlist_allforWC.csv', encoding="utf_8_sig", index=False)


