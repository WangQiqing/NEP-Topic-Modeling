
import re


class Fun_statis():
    def __init__(self):
        pass

    def issuersplit(self, issuelist):
        """
        split policy bodies for each policy text
        :param issuelist:
        :return:
        """
        pattern = r'广东省| |，'
        sep_issuer = []
        for it in issuelist:
            temp = re.split(pattern, it)
            sep_issuer.extend([w for w in temp if w != ''])

        return sep_issuer

    def bodySimplify(self, bodydict):
        issuerdic, newsimpissuer = {},{}
        fp = open('D:/3policyAyc/_database/_auxdata/simplified_issuer.txt', 'r', encoding='utf-8')
        for line in fp.readlines():
            if ',' in line:
                first,last = line.split(',')    # first-condition, last-simplified issuer
                issuerdic.update({first:last.strip('\n')})
        fp.close()
        for fir,las in issuerdic.items():
            count = 0
            if '&' in fir:  # 同时满足这几个条件的原始发布者就可以简化为对应的las
                leng = len(fir.split('&'))
                for k,v in bodydict.items():
                    if len(re.findall(re.sub('&','|',fir),k)) == leng:
                        count += v
                        bodydict[k] = 0
                newsimpissuer.update({las:count})
            else:   # 仅需要满足其一即可将发布者简化委对应的las
                for k,v in bodydict.items():
                    if re.search(fir, k):
                        count += v
                newsimpissuer.update({las:count})
        return newsimpissuer




