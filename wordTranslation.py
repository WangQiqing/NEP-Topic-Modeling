"""translation API for an exclusive goal"""
# coding=utf-8

import http.client
import hashlib
import json
import urllib
import random
import os
import re

import time


class wordTranslation():
    def __init__(self):
        pass


    def getEnglishword(self, singleword):
        appid = '20200410000415605'
        secretKey = '350v6YE9mPbMKvAYRnGn'
        httpClient = None
        myurl = '/api/trans/vip/translate'
        fromLang = 'zh'  # 源语言
        toLang = 'en'  # 翻译后的语言
        salt = random.randint(32768, 65536)
        q = singleword  # 一次性翻译整个列表
        sign = appid + q + str(salt) + secretKey
        sign = hashlib.md5(sign.encode()).hexdigest()
        myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(
            q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
            salt) + '&sign=' + sign

        try:
            httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
            httpClient.request('GET', myurl)
            # response是HTTPResponse对象
            response = httpClient.getresponse()
            jsonResponse = response.read().decode("utf-8")  # 获得返回的结果，结果为json格式
            js = json.loads(jsonResponse)  # 将json格式的结果转换字典结构
            dst = str(js["trans_result"][0]["dst"])  # 取得翻译后的文本结果
            # dlist = re.split(r'[ ]?\\[ ]?[nN][ ]?', dst)
            return dst
        except Exception as e:
            print('err:' + e)
        finally:
            if httpClient:
                httpClient.close()

    def transch2enDict(self, chFreqdic={}):
        """对词频字典转为英文版"""
        kwpath = 'D:/3policyAyc/_database/_auxdata/kwordch2en.txt'
        rawch2enDict = {}
        if os.path.exists(kwpath):
            rawch2enDict.update(self.open_dict(kwpath))  # 获取字典

        enFreqdict = {} # 新的英文-频率词典
        count = 0
        wordlen = len(chFreqdic)
        try:
            for k,v in chFreqdic.items():
                if k in rawch2enDict.keys():    # 如果原始词典有这个中文翻译
                    enFreqdict.update({rawch2enDict.get(k):v})
                else:   # 否则调用翻译接口，同时更新原始字典
                    enword = self.getEnglishword(k)
                    count += 1
                    enFreqdict.update({enword:v})
                    rawch2enDict.update({k:enword})
                    print('\r新词翻译中：.*{}'.format(int(count%6)), end='')  # 动态打印
                    # time.sleep(0.3)  # 休眠1秒
        except:
            self.save_dict(kwpath, rawch2enDict)
            print('翻译出现错误，字典已保存，新增互译词{}个。'.format(count))
        if count>0:
            self.save_dict(kwpath, rawch2enDict)
            print('文件保存完毕，新增英汉互译词{}个。'.format(count))
        return enFreqdict


    def save_dict(self, fname, dictionary):
        js = json.dumps(dictionary)
        file = open(fname, 'w')
        file.write(js)
        file.close()

    def open_dict(self, txtfname):
        file = open(txtfname, 'r')
        js = file.read()
        dic = json.loads(js)
        return dic

    # def baidu_translate(self, contentlist):
    #     appid = '20200410000415605'
    #     secretKey = '350v6YE9mPbMKvAYRnGn'
    #     httpClient = None
    #     myurl = '/api/trans/vip/translate'
    #     fromLang = 'zh'  # 源语言
    #     toLang = 'en'  # 翻译后的语言
    #     salt = random.randint(32768, 65536)
    #     dlist = []  # 翻译后的列表
    #     for item in contentlist:
    #         q = item  # 一次性翻译整个列表
    #         sign = appid + q + str(salt) + secretKey
    #         sign = hashlib.md5(sign.encode()).hexdigest()
    #         myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(
    #             q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
    #             salt) + '&sign=' + sign
    #
    #         try:
    #             httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
    #             httpClient.request('GET', myurl)
    #             # response是HTTPResponse对象
    #             response = httpClient.getresponse()
    #             jsonResponse = response.read().decode("utf-8")  # 获得返回的结果，结果为json格式
    #             js = json.loads(jsonResponse)  # 将json格式的结果转换字典结构
    #             dst = str(js["trans_result"][0]["dst"])  # 取得翻译后的文本结果
    #             dlist.append(dst)
    #         except Exception as e:
    #             print('err:' + e)
    #         finally:
    #             if httpClient:
    #                 httpClient.close()
    #     return dlist


#
# def writeToTxt(list_name, file_path):
#     try:
#         # 这里把翻译结果写入txt
#         fp = open(file_path, "a+", encoding='utf-8')
#         l = len(list_name)
#         i = 0
#         # fp.write('[')
#         for item in list_name:
#             fp.write(str(item))
#             if i < l:
#                 fp.write(',\n')
#             i += 1
#         # fp.write(']')
#         fp.close()
#     except IOError:
#         print("fail to open file")
#
#
# def openfile(fileinput):
#     list = []  ## 空列表
#     # count=1
#     with open(fileinput, "r", encoding='utf-8') as fl:
#         for line in fl.readlines():
#             # print(line)
#             # l=len(line)
#             # print(line[l-3:l-1])
#             # dst='hello world'
#             if line.strip() == '':  # 跳过空行 isspace
#                 print('string is null')
#                 continue
#
#             dst = baidu_translate(line)
#             list.append(dst)
#             time.sleep(2)  # 不要频繁访问百度翻译，睡2秒
#         # count+=1
#         # str='\"'+line[0:l-2]+'\" = \"'+dst+'\"\n'
#         # print(str)
#         # if count==4:
#         #	break
#     # print(list)
#     fileout = r"f:\Users\Desktop\翻译\iOS\dst.txt"
#     writeToTxt(list, fileout)