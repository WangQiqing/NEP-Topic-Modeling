"""
Extract policy texts from MS word files:
1.duplication removing
2.metadata extraction [title, year, issuer, ptext]
3.save to csv files
"""
import docx2txt, re, os
from win32com import client as wc


class Func_extraction():
    def __init__(self):
        pass

    def getBailutext(self, filename):
        """
        get text from word files downloaded on http://www.bailuzhiku.com/
        :param filename: file path of a .docx file
        :return: a dict of structured policy data
        """
        # doc = docx.Document(filename)
        text = docx2txt.process(filename)   # convert to pure text
        fulltext = re.split('[\n]+',text)
        re_chinese = re.compile(u'[\u4e00-\u9fa5]', re.UNICODE)  # Chinese character
        re_num = re.compile(r'20\d{2}')
        textDict, contxt = {'title': '', 'year': '', 'issuer': '', 'ptext': ''}, []
        # dictionary format，contxt- policy text data
        title = filename.split('白鹿智库—')[1].rstrip('.docx')  # get title
        textDict.update({'title': title})

        # tb = doc.tables
        # for para in doc.paragraphs:
        #     fulltext.append(para.text)

        for i, itxt in enumerate(fulltext):
            if "发文机构：" in itxt:
                tmp = itxt.split('：')
                match = re.search(re_chinese, tmp[1])
                if match:
                    textDict.update({'issuer': tmp[1]})  # get issuer
                else:
                    textDict.update({'issuer': ''})
            if "发文字号：" in itxt:
                tmp = itxt.split('：')
                match = re.search(re_num, tmp[1])
                if match:
                    textDict['year'] = match.group()    # try to get release date
            if "发布日期：" in itxt:
                tmp = itxt.split('：')
                match = re.search(re_num, tmp[1])
                if match:
                    textDict['year'] = match.group()    # try to get release date
            if "失效日期：" in itxt:
                contxt = [sent for sent in fulltext[i+1:] if '白鹿智库' not in sent]
                textDict.update({'ptext': '\n'.join(contxt)})  # get raw text
                break
        return textDict


    def getFabaotext(self, filename):
        """
        get text from word files downloaded on https://www.pkulaw.com/
        :param filename: file path
        :return: a dict of policy text
        """
        # 获取从北大法宝网站下载的文档文本
        # doc = docx.Document(filename)
        doc = docx2txt.process(filename)
        fulltext = re.split('[\n]+',doc)
        re_chinese = re.compile(u'[\u4e00-\u9fa5]', re.UNICODE)
        re_num = re.compile(r'20\d{2}')
        re_bra = re.compile(r'\(FBM.*\)')

        textDict, contxt = {'title':'','year':'','issuer':'','ptext':''},[]
        title = re.sub(r'\(FBM.*\)', '', filename)
        title = title.split('北大法宝\\')[1].strip('.docx')
        textDict.update({'title': title})
        #
        # for para in doc.paragraphs:
        #     fulltext.append(para.text)
        ind_start, ind_end = 0, 0
        for i, itxt in enumerate(fulltext):
            if "发布部门" in itxt and ":" in itxt:
                tmp = itxt.split(':')
                match = re.search(re_chinese, tmp[1])
                if match:
                    textDict.update({'issuer': tmp[1]})
                else:
                    textDict.update({'issuer': ''})
            if "发文字号" in itxt and ":" in itxt:
                tmp = itxt.split(':')
                match = re.search(re_num, tmp[1])
                if match:
                    textDict['year'] = match.group()
            if "发布日期" in itxt and ":" in itxt:
                tmp = itxt.split(':')
                match = re.search(re_num, tmp[1])
                if match:
                    textDict['year'] = match.group()

            if "法规类别" in itxt or "效力级别" in itxt:
                ind_start = i + 1
            if '本篇引用的法规' in itxt:
                ind_end = i
                break
            if '北大法宝' in itxt:
                ind_end = i

        if ind_end == 0:
            contxt.extend(fulltext[ind_start:])
        else:
            contxt.extend(fulltext[ind_start:ind_end])
        textDict.update({'ptext': '\n'.join(contxt)})
        return textDict

    def getFabaohtml(self, filename):
        """
        get Fabao text in html format
        """
        with open(filename, 'r', encoding='utf-8') as fp:
            htext = fp.read()
        soup = BeautifulSoup(htext, 'html.parser')
        textDict = {'title':'','year':'','issuer':'','ptext':''}
        # 获取标题
        title = soup('h2', 'title')
        textDict['title'] = title[0].text.strip()
        # 获取发布部门和日期
        divfield = soup.find('div', attrs='fields')
        for tag in divfield.ul.children:
            if isinstance(tag, bs4.element.Tag):
                strongtext = tag.find('strong').text
                if '发布部门' in strongtext:
                    textDict['issuer'] = tag.find('span')['title']
                elif '发布日期' in strongtext:
                    textDict['year'] = re.findall('2\d{3}', tag.text)[0]
        # 获取正文
        divfulltext = soup.find('div', 'fulltext')
        text = []
        for tag in divfulltext.children:
            if isinstance(tag, bs4.element.NavigableString):
                text.append(tag.string)
            else:
                text.append(tag.text)
        contxt = [str(it).replace('\n','') for it in text if it != '' and it != '\n']
        textDict['ptext'] = '\n'.join(contxt)
        return textDict

    def getdocPaths(self, rootpath):
        """trans .doc to .docx, return all filepaths"""
        if not os.path.exists(rootpath):
            print('No file available, please check the rootpath.')
            return
        filepaths,count = [],0  # filepaths of policy texts(.docx format)
        fnls = [fn for fn in os.listdir(rootpath) if not fn.startswith('~$')]
        for name in fnls:    # filename list in the given file path
            fn = os.path.join(rootpath, name)
            if fn.endswith('.doc'):
                if not os.path.exists(fn+'x'):     # trans doc to docx
                    new_name = fn + 'x'
                    try:
                        w = wc.Dispatch('Word.Application')
                        w.visible = 0
                        doc = w.Documents.Open(fn)
                        doc.SaveAs2(new_name, FileFormat=16)
                        doc.Close()
                        filepaths.append(new_name)
                        count += 1
                        print('第{}个Word文件转换成功！'.format(count))
                    except:
                        print("该文件处理失败，可能有损坏："+fn)
            elif fn.endswith('.docx') and fn not in filepaths:
                filepaths.append(fn)
            elif fn.endswith('.html'):
                filepaths.append(fn)
        return filepaths
