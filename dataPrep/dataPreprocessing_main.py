"""main file of data extraction and chunking"""

import unicodedata, re, os
from dataPrep.Func_extraction import Func_extraction
from dataPrep.Func_chunking import Func_chunking
import pandas as pd
from timeit import default_timer as timer


"""start extraction"""
extractor = Func_extraction()
chunker = Func_chunking()

provnames = {'广东':1,'江苏':2,'山东':3,'四川':4,'内蒙古':5,'新疆':6, '国家':7}
print('Which province\'s policy texts do you want to extract:\n1-广东\n2-江苏\n3-山东\n4-四川\n5-内蒙古\n6-新疆\n7-国家')
namenum = int(input('>>>>>Choose a number:'))
fnames_bailu, fnames_fabao = [],[]
rrootp = r'D:\2020止水禪心\0.3新能源政策挖掘\数据收集'
for it in provnames:
    if namenum == provnames.get(it):
        rootpath_bailu = rrootp+'\\文本-'+it+'\\白鹿智库'
        rootpath_fabao = rrootp + '\\文本-' + it + '\\北大法宝'
        print('路径获取中......')
        fnames_bailu = extractor.getdocPaths(rootpath_bailu)
        fnames_fabao = extractor.getdocPaths(rootpath_fabao)
        print('路径获取完毕！白鹿智库有效文件{}个，北大法宝有效文件{}个'.format(len(fnames_bailu),len(fnames_fabao)))
        break

print('开始抽取白鹿智库政策文本......')
nullpath, extractedTXT = [],[]   # text list
for fpath in fnames_bailu:    #抽取白鹿智库文本
    tmp = extractor.getBailutext(fpath)
    if tmp['year']=='' or tmp['issuer']=='' or tmp['ptext']==[]:
        print(fpath+"这个文件异常！！")
        nullpath.append(fpath)
    else:
        extractedTXT.append(tmp)
print('白鹿智库文本抽取完毕！')

print('开始抽取北大法宝政策文本......')
for i, fpath in enumerate(fnames_fabao):    #抽取北大法宝文本
    if fpath.endswith('.docx'):
        tmp = extractor.getFabaotext(fpath)
    elif fpath.endswith('.html'):
        tmp = extractor.getFabaohtml(fpath)
    if tmp['year'] =='' or  tmp['ptext']=='' or tmp['issuer']=='':
        print(fpath+"这个文件异常！！")
        nullpath.append(fpath)
    else:
        extractedTXT.append(tmp)
print('所有文档抽取完毕！！！共有'+str(len(extractedTXT))+'个文档。')
raw_df = pd.DataFrame(extractedTXT)

print('开始执行去重操作......')
"""enhancing dulplicates removing"""
#日期是否一致，标题相似度
raw_df.drop_duplicates()
titls = raw_df['title'].tolist()
yearls = raw_df['year'].tolist()
duptupls = []   # 元组列表，可能相似的两篇索引
leng = len(titls)
for i in range(0, leng):
    for j in range(i+1, leng):
        if yearls[i]==yearls[j]:
            temptiti = chunker.clear_character(titls[i])
            temptitj = chunker.clear_character(titls[j])
            if temptiti == temptitj:   # 名称相同且年份相同则为重复项
                duptupls.append((i,j))
            else:
                break
        else:
            break
if duptupls:
    print('重复数据{}项，查询并删除......'.format(len(duptupls)))
    dupinds = [tup[1] for tup in duptupls]  # 获取要删除的索引
    raw_df = raw_df.drop(dupinds)
else:
    print('无重复数据')

savpath = os.path.abspath(r'D:\3policyAyc\_database\_policytxt\Raw_'+list(provnames.keys())[namenum-1]+'1.csv')
raw_df.to_csv(savpath, encoding="utf_8_sig", index=False)
print('保存完毕，有效数据共{}条！'.format(raw_df.shape[0]))


"""start preprocessing"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
province = input('输入需要分词的省份：')
raw_df = pd.read_csv(r'D:\3policyAyc\_database\_policytxt\Raw_'+province+'.csv')   # for test
size = raw_df.shape
ptlist = []     # list of each policy text
for i in range(0, size[0]):
    temp = raw_df['ptext'].tolist()[i]  # policy text
    temp1 = unicodedata.normalize('NFKC', temp) # handle \u3000,\ax0
    temp2 = ''.join(temp1.split(' '))
    textlist = temp2.split('\n')
    for it in textlist.copy():
        if not it:
            textlist.remove('') # remove ''
    ptlist.append(textlist)   # each element is a list

"""remove the element containing rmvs"""
print('正在删除无关文本和片段'+'>>>>'*7)
tic = timer()
newptlist = []
for i, entrylis in enumerate(ptlist):
    temp = []
    title = raw_df['title'][i]
    for ele in entrylis:
        if not chunker.containRmvs(title, ele):
            temp.append(ele)    # exclude elements with rmvwords
    newptlist.append(temp)  # relatively clean, but contains unrelated strings

"""remove unrelated strings """
pattern1 = re.compile('^各.*[:：]$')
deleind = []
for i,entrylis in enumerate(newptlist):
    for s in entrylis.copy():
        if chunker.chinesechars(s) <= 10:
            entrylis.remove(s)
        elif re.match(pattern1, s):
            entrylis.remove(s)
    if entrylis == []:
        deleind.append(i)
toc = timer()
print('删除完毕，用时' + str(toc - tic) +'秒！')
raw_df['ptext'] = newptlist
df = raw_df.drop(index=deleind, axis=0)   # delete rows with null ptlist
# df.to_excel('../_database/_policytxt/test1.xlsx')

# """attempt to chunk sentences for other usage"""
"""word segmentation"""
print('开始执行去除停用词和分词操作'+'>>>>'*7)
tic1 = timer()
newptlist = df['ptext'].tolist()
ptwordlist = []
for entrylis in newptlist:
    new_entry = []
    for sent in entrylis:
        new1 = chunker.big2small_num(sent)
        new2 = chunker.upper2lower(new1)
        new3 = chunker.clear_character(new2)
        new_entry.append(new3)

    words = '\n'.join(chunker.clean_cut_jieba(new_entry))
    ptwordlist.append(words)
df['ptext'] = ptwordlist
toc1 = timer()
print('分词完毕！用时'+str(toc1-tic1)+'秒！')

# df.to_excel(r'D:\3policyAyc\_database\_policytxt\Wordlist_'+list(provnames.keys())[namenum-1]+'.xlsx')
df.to_csv(r'D:\3policyAyc\_database\_policytxt\Wordlist_'+province+'0.csv', encoding="utf_8_sig", index=False)