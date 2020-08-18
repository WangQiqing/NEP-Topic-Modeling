# coding:utf-8
"""
using TF-IDF to visualize word distribution with wordCloud
"""

from PIL import Image
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from gensim import corpora
from wordTranslation import wordTranslation
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签


def generate_wordfreq(ptlist, ft):
    """
    preprocess, including rare words removing and keywords selection
    ptlist: a list of lists
    :return: word frequency dict
    """
    newptlist, stopwords = [],[]
    # fp = open('D:/3policyAyc/_database/_auxdata/rmvs_for_wordcloud.txt', 'r', encoding='utf-8')
    # for line in fp.readlines():
    #     if line != '' and line != '\n':
    #         stopwords.append(line.strip('\n'))
    # stopwords = list(set(stopwords))
    for doc in ptlist:
        temp = [w for w in doc if len(w)>1]
        newptlist.append(temp)
    dictionary = corpora.Dictionary(documents=newptlist, prune_at=None)
    dictionary.filter_extremes(no_below=100, no_above=ft, keep_n=None)#过滤掉出现次数少于nobelow的，保留在不超过no_above*100%的文档中都出现的词
    tokenls = list(dictionary.values()) # 索引即为单词编码
    freqdic = {}
    for k,v in dictionary.dfs.items():
        freqdic.update({tokenls[k]:v})
    print('词频生成成功，共{}个独特词。'.format(len(freqdic)))
    return freqdic


def generate_wordcloud(wordfreq, ft):
    '''
    输入文本生成词云,如果是中文文本需要先进行分词处理
    '''
    # 设置显示方式
    #d=path.dirname(__file__)
    a_mask = np.array(Image.open("D:/3policyAyc/_database/_auxdata/fontImage//图片椭圆.jpg"))
    font_path = "D:/3policyAyc/_database/_auxdata/fontImage//msyh.ttf"
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color="white",# 设置背景颜色
           max_words=1500, # 词云显示的最大词数
           mask=a_mask,# 设置背景图片
           stopwords=stopwords, # 设置停用词
           font_path=font_path, # 兼容中文字体，不然中文会显示乱码
                   max_font_size=60
                  )
    # 生成词云
    wc.generate_from_frequencies(wordfreq)
    # 生成的词云图像保存到本地
    wc.to_file( "D:/3policyAyc/_database/_interresults/词云图"+str(round(ft,2))+".png")
    print('保存参数no_above={}时生成的词云图'.format(round(ft,2)))
    # 显示图像
    # plt.imshow(wc, interpolation='bilinear')
    # # interpolation='bilinear' 表示插值方法为双线性插值
    # plt.axis("off")# 关掉图像的坐标
    # plt.show()




if __name__ == '__main__':
    df = pd.read_csv('D:/3policyAyc/_database/_policytxt/Wordlist_allforWC.csv')
    ptls = [doc.split('\n') for doc in df['ptext'].tolist()]

    wordtrans = wordTranslation()

    # 确定最佳no_above参数值
    paras = np.arange(1, 0, -0.05)
    unicwords = []

    for para in paras:
        temp = generate_wordfreq(ptls, para)
        unicwords.append(len(temp))
        enfreqdict = wordtrans.transch2enDict(temp)
        df = pd.DataFrame(data=enfreqdict, index=['0'])
        df.to_excel('D:/3policyAyc/_database/_interresults/Wordfreqdic' + str(round(para,2)) + '.xlsx')
        print('过滤参数no_above={}时生成的词频字典已保存'.format(para))

    xaxislabel = np.array([str(round(i,2)) for i in paras])
    xdata = xaxislabel[np.argsort(-paras)]
    ydata = np.array(unicwords)[np.argsort(-paras)]
    plt.xticks(rotation=270)
    plt.plot(xaxislabel, ydata, color='blue',linewidth=2.0, markersize=8.0, linestyle='--', marker='^')
    tmpdf = pd.DataFrame({'xdata':xdata,'ydata':ydata})
    tmpdf.to_excel('D:/3policyAyc/_database/_interresults/Word_count_on_noabove.xlsx')


    # 词频字典已存在
    paras = np.arange(1, 0, -0.05)
    for para in paras:
        df = pd.read_excel('D:/3policyAyc/_database/_interresults/Wordfreqdic'+str(para)+'.xlsx')
        colls = list(df.columns.values)
        ls = list(df.loc[0])
        enwordfreq = dict(zip(colls[1:], ls[1:]))
        # print('mostfreqworis:{},itsfreqis:{}'.format(enwordfreq.get(max(enwordfreq.values())),max(enwordfreq.values())))
        generate_wordcloud(enwordfreq, para)

    # texts = [['这是', '一个', '文本'], ['这是', '第二个', '文本'], ['这是', '又一个', '文本'], ['这是', '最后', '一个', '文本']]
    # dictionary = corpora.Dictionary(texts)
    # corpus = [dictionary.doc2bow(text) for text in texts]
    # tf_idf_model = TfidfModel(corpus, normalize=False)
    # word_tf_tdf = list(tf_idf_model[corpus])
    # print('词典:', dictionary.token2id)
    # print('词频:', corpus)
    # print('词的tf-idf值:', word_tf_tdf)

    #
    # chunksize = 10 # 分批次翻译
    # leng = len(chwords)
    # chunknum = math.floor(leng / chunksize)
# for j in range(0, math.floor(leng / chunksize) + 1):
#     if chunksize * (j + 1) <= leng:
#         temp = wordtrans.baidu_translate(chwords[chunksize * j:chunksize * (j + 1)])
#         # temp = chwords[chunksize*j:chunksize*(j+1)]
#         enwords.extend(temp)
#     else:
#         temp = wordtrans.baidu_translate(chwords[chunksize * j:])
#         # temp = chwords[chunksize*j:]
#         enwords.extend(temp)