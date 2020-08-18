"""Simple statistical analysis of the total extracted policy
text, detecting basic temporary and spatial distribution"""
import re, collections, os
import pandas as pd
from statAyc.Func_statis import Fun_statis
from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora

# 读取所有政策文本数据至一个Dataframe
dflist = []
rpath = r'D:/3policyAyc/_database/_policytxt'
provls = []
for name in os.listdir(r'D:/3policyAyc/_database/_policytxt'):
    if 'Wordlist' in name and name.endswith('csv'):
        provls.append(name)
        tempdf = pd.read_csv(os.path.join(rpath, name))
        tempdf = tempdf.drop(index=(tempdf.loc[(tempdf['year'] <= 2009)].index))
        tempdf = tempdf.drop(index=(tempdf.loc[(tempdf['year'] >= 2020)].index))  # 保留2010-2019
        print(tempdf.shape)
        dflist.append(tempdf)
# dfdict = dict(zip(provls, dflist))
df = pd.concat(dflist)
df.to_csv(r'D:\3policyAyc\_database\_policytxt\Wordlist_all0.csv', encoding="utf_8_sig", index=False)

fun_statis = Fun_statis()   # initialize the class
# df_nation = pd.read_excel(r'./txtdata/nationalPolicy.xlsx')
# df_guangdong = pd
# df.jiangsu = pd
# df1 = df[1:5]
# df3 = df1.drop(index=df1.loc[["通知" in x for x in df1['title'].values.tolist()]].index)
# df3 = df1.drop(index=df2.index)
# df_check = df.loc[["目录" in x for x in df['title'].values.tolist()]]  # 取索引为2的行
"""1.政策发布时间分布，2010-2019的文本"""
yearlist = df['year'].tolist()
syear = pd.Series(yearlist)
countDict = dict(syear.value_counts())  # frequency calculation
temp_df1 = pd.DataFrame(data=countDict, index = ['Count'])
proportionDict = dict(syear.value_counts(normalize=True))
temp_df2 = pd.DataFrame(data=proportionDict, index = ['Frequency'])
year_df = pd.concat([temp_df1,temp_df2])
year_df.to_excel('./_database/_interresults/stat_by_year.xlsx')


"""2.政策发布机构分析"""
bodies = fun_statis.issuersplit(df['issuer'].tolist())  # extract policy issuers
bodyDic = pd.Series(bodies).value_counts()
tempbodydf = pd.DataFrame(data=bodyDic)
tempbodydf.to_excel('D:/3policyAyc/Rawpolicybodies.xlsx')

# simplify and merge similar bodies
simpbodyDict = fun_statis.bodySimplify(bodyDic)
df_body = pd.DataFrame(data=simpbodyDict, index=['Count'])
df_body.to_excel('D:/3policyAyc/_database/_interresults/stat_by_issuer.xlsx') # 测试，找出发布频率高的主体

"""3.词频分布统计"""
wordlist, ptextlist = [],[]
for entry in df['ptext'].tolist():
    temp = entry.split('\n')
    ptextlist.append(temp)
    wordlist.extend(temp)
# 统计词频
wordfreq = collections.Counter(wordlist)
mostfreqlist = wordfreq.most_common(100)    # 查看出现频率最高的n个词
tempk, tempv = [],[]
for tup in mostfreqlist:
    tempk.append(tup[0])
    tempv.append(tup[1])
mostfreqdic = {'Word': tempk, 'Count': tempv}
df_mostfreq = pd.DataFrame(mostfreqdic)
# wordfreq = wordseries.value_counts(normalize=True)
df_mostfreq.to_excel('D:/3policyAyc/_database/_interresults/wordfreq_distribution.xlsx')

# tf-idf
# ptextDict = corpora.Dictionary(ptextlist)
# corpus_ptext = [ptextDict.doc2bow(w) for w in ptextlist]    # doc2bow(1,2)表示第*篇文档中编号1的单词出现2次
# tf_idf_model = TfidfModel(corpus_ptext, normalize=True)
# word_tf_idf = list(tf_idf_model[ptextDict.token2id])
# count = 0
# for it in word_tf_idf:
#     count+= len(it)
# print(count)
# sorted_words = sorted(b.items(), key=lambda x: x[1], reverse=True)
# tfidf_df = pd.DataFrame([(tup[0],tup[1]) for a in word_tf_idf for tup in a], columns=['Word','Code'])
# tokenid_df = pd.DataFrame([(k,v) for k,v in ptextDict.items()], columns=['Code', 'TF-IDFvalue'])
# tf_idfDic = {}  # 结果转为字典形式
# for it in word_tf_idf:
#     for tup in it:
#         tf_idfDic.update({tup[0]:tup[1]})
# tokenid_df = pd.DataFrame(data=ptextDict.token2id, columns=['Word','Code'])
# tokenid_df['Code'].dtype
# tfidf_df = pd.DataFrame(data=tf_idfDic, index=['Code', 'TF-IDFvalue'])
# tfidf_df['Code'].dtype
#
# wordtfidf
# DDI = cordi.id2token
#
#
#
# organs = []
# pat = re.compile(r'/|[ ]+|、|,|//|，')
# pat1 = re.compile(r'[\\u3000]+|[\u3000]+|[\xa0]+|[?]+')
# for it in organlist:
#     if isinstance(it,str):
#         if re.search(pat1,it):
#             tmp = ''.join(re.split(pat1,it))
#             organs.extend(re.split(pat,tmp))
#         else:
#             organs.extend(re.split(pat,it))
#
# from collections import Counter
# orgcount = Counter(organs)
# orgcount.pop('')
# orgcount.pop('建设部(已撤销)')
# for k,v in orgcount.items():
#     if '委员会' in k:
#         k_new = k.replace('委员会','委')
#         orgcount[k_new] = orgcount.pop(k)
# orgcount
#
# def getMatch(pat, sstr = '财政部科技部工业和信息化部发展改革委'):
#     '''获取所有匹配的部分及其索引，返回tuple_list
#     pat:正则匹配表达式
#     sstr:待搜索的字符串
#     '''
#     if re.search(pat,sstr):
#         all_ind = []
#         all_find = re.findall(pat,sstr)
#         for it in all_find:
#             all_ind.append(sstr.index(it))
#             sstr = sstr.replace(it,'爨',1)
#         mat_tuple = tuple(zip(all_find,all_ind))
#         return list(mat_tuple)
#     else:
#         print('Nothing had been matched to the given string!')
#         return
#
# pat2 = '部|局|委'
# div_list = []
# for k,v in list(orgcount.items()):
# #     count.append(v)
# # ss = pd.Series(count)
# # ss.value_counts()
#     if len(re.findall(pat2,k))>=2:
#         matlist = getMatch(pat2,k)
#         tmp_dic = {}
#         for i,tup in enumerate(matlist):
#             new_key = ''
#             if i == 0:
#                 new_key = k[0:tup[1]+1]
#             else:
#                 new_key = k[matlist[i-1][1]+1:tup[1]+1]
#             tmp_dic.update({new_key:v})
#         div_list.append(tmp_dic)
#
# for dic in div_list:
#     for dic_k in dic.keys():
#         if dic_k in orgcount:
#             td = {dic_k:dic.get(dic_k)+orgcount.get(dic_k)}
#             orgcount.pop(dic_k)
#             orgcount.update(td)
#         else:
#             orgcount.update({dic_k:dic.get(dic_k)})
#
# for k,v in list(orgcount.items()):
#     if len(re.findall(pat2,k))>=2:
#         orgcount.pop(k)
# orgcount
# org_bac = orgcount.copy()
# orgcount = org_bac.copy()
#
# for k in list(orgcount.keys()):
#     tl = re.split(pat2,k)
#     if len(tl)>1 and tl[1]:
#         tl[0] = tl[0]+re.search(pat2,k).group()
#         if tl[0] in orgcount:
#             td = {tl[0]:orgcount.get(k)+orgcount.get(tl[0])}
#             orgcount.pop(k)
#             orgcount.pop(tl[0])
#             orgcount.update(td)
#         else:
#             orgcount[tl[0]] = orgcount.pop(k)
#
# df_organ = pd.DataFrame.from_dict(orgcount, orient='index',columns=['pieces'])
# df_organ = df_organ.reset_index().rename(columns = {'index':'organization'})
# df_organ.to_excel('./_interresults/by_organ1.xlsx')
