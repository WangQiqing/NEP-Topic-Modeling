"""Sentence&Word level preprocessing
1.Remove unrelated elements in extractedTXT
2.Rightly chunk the words and save them into another excel file
"""
import re, jieba
import jieba.posseg as pseg

class Func_chunking():
    def __init__(self):
        pass

    def containRmvs(self, title, slicetext=''):
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
            fp = open('D:/3policyAyc/_database/_auxdata/rmvs_sentchunk.txt', 'r', encoding='utf-8')
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

    def chinesechars(self, string):
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

    def big2small_num(self, sentence):  # 应当考虑某些包含了大写数字的词语
        numlist = {"十一五":"onefive","十二五":"twofive","十三五":"threefive","十四五":"fourfive", "珠三角":"zhusanjiao",
                   "一体化": "yitihua", "长三角":'changsanjiao', "四川":"sichuan", "三峡":"sanxia",
                   "一":"1","二":"2","三":"3","四":"4","五":"5","六":"6","七":"7","八":"8","九":"9","零":"0"}
        for item in numlist:
            sentence = sentence.replace(item, numlist[item])
        reclist = {"十一五":"onefive","十二五":"twofive","十三五":"threefive","十四五":"fourfive", "珠三角":"zhusanjiao",
                   "一体化": "yitihua", "长三角":'changsanjiao', "四川":"sichuan", "三峡":"sanxia"}
        for k,v in reclist.items():
            sentence = sentence.replace(v, k)
        return sentence

    def upper2lower(self, sentence):
        new_sentence=sentence.lower()
        return new_sentence

    def clear_character(self, sentence):
        """get rid of unrelated character"""
        pattern1='[a-zA-Z0-9]'
        pattern2 = r'\[.*?\]'
        pattern3 = re.compile(u'[^\\s1234567890:：' + '\u4e00-\u9fa5]+')
        pattern4='[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
        line1=re.sub(pattern1,'',sentence)   #去除英文字母和数字
        line2=re.sub(pattern2,'',line1)   #去除表情
        line3=re.sub(pattern3,'',line2)   #去除其它字符
        line4=re.sub(pattern4, '', line3) #去掉残留的冒号及其它符号
        new_sentence=''.join(line4.split()) #去除空白
        return new_sentence

    def clean_cut_jieba(self, contents):
        """
        remove stopwords and segment words with jieba
        :param contents:text pieces list for a policy doc
        :param use_diy: using diy stopwords or not
        :return: new list without stopwords
        """
        exstopwords=[]
        stopwords = {}.fromkeys([line.rstrip() for line in open('D:/3policyAyc/_database/_auxdata/stop_words', encoding="utf-8")]) #读取停用词表
        stopwords_list = set(stopwords)
        fp = open('D:/3policyAyc/_database/_auxdata/ex_stopwords.txt','r',encoding='utf8')
        for line in fp.readlines():
            exstopwords.append(line.strip('\n'))
        stopwords_list.update(exstopwords)
        fp.close()
        jieba.load_userdict('D:/3policyAyc/_database/_auxdata//userdict.txt')  # 导入用户自定义词典
        words_list = [w for w in jieba.cut(''.join(contents), cut_all=False) if w not in stopwords_list] # cut_all是
        # 分词模式，True是全模式，False是精准模式，默认False
        return words_list

    def clean_cut_pkuseg(self, contents):
        """
        remove stopwords and segment words with pkuseg
        :param contents:text pieces str for a policy doc
        :param use_diy: using diy stopwords or not
        :return: new list without stopwords
        """
        exstopwords=[]
        stopwords = {}.fromkeys([line.rstrip() for line in open('D:/3policyAyc/_database/_auxdata/stop_words', encoding="utf-8")]) #读取停用词表
        stopwords_list = set(stopwords)
        fp = open('D:/3policyAyc/_database/_auxdata/ex_stopwords.txt','r',encoding='utf8')
        for line in fp.readlines():
            exstopwords.append(line.strip('\n'))
        stopwords_list.update(exstopwords)
        fp.close()
        seg = pkuseg.pkuseg(user_dict = "D:/3policyAyc/_database/_auxdata//userdict.txt")
        words_list = [word for word in seg.cut(''.join(contents)) if word not in stopwords_list]

        return words_list

    def cutwithPOS(self, rawstrtext, remainvn=True):
        """
        并行分词并进行词性标注，返回单篇文档中的名词n，动词v，形容词a和专有名词n,并去除停用词
        :param rawstrtext: 整篇政策文档字符串
        :param remainvn:是否保留除vn外的其他动词
        :return: [w1,w2,...]w为提取出来的n,v,a
        """
        new1 = self.big2small_num(rawstrtext)
        new2 = self.upper2lower(new1)
        newcont = self.clear_character(new2)
        exstopwords, wordlist = [],[]
        stopwords = {}.fromkeys(
            [line.rstrip() for line in open('D:/3policyAyc/_database/_auxdata/stop_words', encoding="utf-8")])  # 读取停用词表
        stopwords_list = set(stopwords)
        fp = open('D:/3policyAyc/_database/_auxdata/ex_stopwords.txt', 'r', encoding='utf8')    # 自定义停用词
        for line in fp.readlines():
            exstopwords.append(line.strip('\n'))
        stopwords_list.update(exstopwords)
        fp.close()
        jieba.load_userdict('D:/3policyAyc/_database/_auxdata//userdict.txt')  # 导入用户自定义词典
        wordsiter = pseg.cut(newcont)
        pat = re.compile('[na]|vn')
        if not remainvn:
            pat = re.compile('[nav]')
        for w in wordsiter:
            temp = w.word
            if re.search(pat, w.flag) and temp not in stopwords_list:
                wordlist.append(temp)

        return wordlist

    def removeProvs(self, wordlist):
        """
        remove province and city names
        :param wordlist: a list of words
        :return: list of words without any provname or cityname
        """
        fp = open('D:/3policyAyc/_database/_auxdata//rmvs_provandcity.txt','r')
        stopwords = [line.strip('\n') for line in fp.readlines()]
        fp.close()
        cleanedwls = wordlist.copy()
        count, cmax = 0, len(wordlist)
        for word in wordlist:
            count += 1
            for stw in stopwords:
                if word in stw:
                    cleanedwls.remove(word)
                    print('\r正在删除省份和城市名:{:2f}%'.format(count*100/cmax), end='')
                    break
        return cleanedwls





