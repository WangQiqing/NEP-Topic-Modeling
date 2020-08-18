"""Sentence&Word level preprocessing
1.Remove unrelated elements in extractedTXT
2.Rightly chunk the words and save them into another excel file
"""
import re, unicodedata, jieba
import pandas as pd
from timeit import default_timer as timer



def containRmvs(title, slicetext=''):
    """
    judge if the slicetext contains words or title in removewords
    :param slicetext: string
    :return: bool
    """
    if slicetext == '':
        print('Got nothing in the slicetext.')
        return False
    else:
        # get stopwords for removing unrelated elements
        fp = open('../_database/_auxdata/rmvs_sentchunk.txt', 'r', encoding='utf-8')
        rmvs = []
        for word in fp.readlines():
            word = word.strip('\n')
            word = word.strip()
            if '&' in word:
                word = word.split('&')
            rmvs.append(word)
        fp.close()
        rmvs.append(title)
        for word in rmvs:
            if isinstance(word, str):
                if word in slicetext:
                    return True
            elif isinstance(word, list):
                count = 0
                for it in word:
                    if it in slicetext:
                        count += 1
                if count == len(word):
                    return True
    return False

def chinesechars(string):
    """calculate how many chinese chars in the string"""
    re_chinese = re.compile(u'[\u4e00-\u9fa5]', re.UNICODE)
    count = 0
    for str in string:
        if re.match(re_chinese, str):
            count += 1
    return count

# def cut_sent(para):
#     """Chinese sentence segmenting"""
#     para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
#     para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
#     para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
#     para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
#     # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
#     para = para.rstrip()  # 段尾如果有多余的\n就去掉它
#     # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
#     return para.split("\n")

def big2small_num(sentence):
    numlist = {"十一五":"onefive","十二五":"twofive","十三五":"threefive","十四五":"fourfive",
               "一":"1","二":"2","三":"3","四":"4","五":"5","六":"6","七":"7","八":"8","九":"9","零":"0"}
    for item in numlist:
        sentence = sentence.replace(item, numlist[item])
    reclist = {"十一五":"onefive","十二五":"twofive","十三五":"threefive","十四五":"fourfive"}
    for k,v in reclist.items():
        sentence = sentence.replace(v, k)
    return sentence

def upper2lower(sentence):
    new_sentence=sentence.lower()
    return new_sentence

def clear_character(sentence):
    """get rid of unrelated character"""
    pattern1='[a-zA-Z0-9]'
    pattern2 = '\[.*?\]'
    pattern3 = re.compile(u'[^\s1234567890:：' + '\u4e00-\u9fa5]+')
    pattern4='[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    line1=re.sub(pattern1,'',sentence)   #去除英文字母和数字
    line2=re.sub(pattern2,'',line1)   #去除表情
    line3=re.sub(pattern3,'',line2)   #去除其它字符
    line4=re.sub(pattern4, '', line3) #去掉残留的冒号及其它符号
    new_sentence=''.join(line4.split()) #去除空白
    return new_sentence

def clean_cut(contents, use_diy = 0):
    """
    remove stopwords
    :param contents:text pieces list for a policy doc
    :param use_diy: using diy stopwords or not
    :return: new list without stopwords
    """
    contents_list, exstopwords=[],[]
    stopwords = {}.fromkeys([line.rstrip() for line in open('D:/3policyAyc/_database/_auxdata/stop_words', encoding="utf-8")]) #读取停用词表
    stopwords_list = set(stopwords)
    if use_diy:
        fp = open('D:/3policyAyc/_database/_auxdata/ex_stopwords.txt','r', encoding='utf8')
        for line in fp.readline():
            exstopwords.append(line)
        stopwords_list.update(exstopwords)
    newwordtlist = []
    jieba.load_userdict('D:/3policyAyc/_database/_auxdata//userdict.txt')  # 导入用户自定义词典
    for row in contents:      #循环去除停用词
        words_list = jieba.cut(row, cut_all=False)  # cut_all是分词模式，True是全模式，False是精准模式，默认False
        newwordtlist.extend(words_list)
        # words = [w for w in words_list if w not in stopwords_list]
        # newwordtlist.extend(words)
    return newwordtlist


if __name__ == '__main__':
    raw_df = pd.read_excel('../_database/_policytxt/Raw_jiangsu.xlsx')
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
        # st = temp2.split('\n')
        # c = 0
        # for it in temp2:
        #     if re.search('', it):
        #         c+=1
        # print(c)


    """remove the element containing rmvs"""
    print('正在删除无关文本和片段'+'>>>>'*7)
    tic = timer()
    newptlist = []
    for i, entrylis in enumerate(ptlist):
        temp = []
        title = raw_df['title'][i]
        for ele in entrylis:
            if not containRmvs(title, ele):
                temp.append(ele)    # exclude elements with rmvwords
        newptlist.append(temp)  # relatively clean, but contains unrelated strings

    """remove unrelated strings """
    pattern1 = re.compile('^各.*[:：]$')
    deleind = []
    for i,entrylis in enumerate(newptlist):
        for s in entrylis.copy():
            if chinesechars(s) <= 10:
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

    """attempt to chunk sentences for other usage"""
    """word segmentation"""
    print('开始执行去除停用词和分词操作'+'>>>>'*7)
    tic1 = timer()
    newptlist = df['ptext'].tolist()
    ptwordlist = []
    for entrylis in newptlist:
        new_entry = []
        for sent in entrylis:
            new1 = big2small_num(sent)
            new2 = upper2lower(new1)
            new3 = clear_character(new2)
            new_entry.append(new3)

        words = '\n'.join(clean_cut(new_entry))
        ptwordlist.append(words)
    df['ptext'] = ptwordlist
    toc1 = timer()
    print('分词完毕！用时'+str(toc1-tic1)+'秒！')
    df.to_excel('../_database/_policytxt/wordlist_Jiangsu.xlsx')

