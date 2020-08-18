
# 面积图--DTM训练结果总图
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#中文及负号处理，字体
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 8,
         }
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 10,
         }

def plotDTMTotal():
    # figure
    fig = plt.figure(figsize=(6, 6))
    x = np.arange(2010, 2020, dtype=int)  # x坐标年份
    # data
    for i in range(1,7):
        sheetname = 'Sheet'+str(i)
        tmpdf = pd.read_excel(r'D:\tmpWORKSTATION\testdtm.xlsx', sheet_name=sheetname)
        y = tmpdf.transpose().values
        ax = fig.add_subplot(2,3,i)
        labels = list(tmpdf.keys())
        plt.xlim(2010, 2019)  # 设置x的范围
        ax.stackplot(x, y, labels=labels)  # 堆积面积图
        ax.set_xticks(range(2010, 2020, 3)) # 横坐标标签
        plt.xlabel('', fontdict=font2)  # 横纵坐标
        plt.ylabel('', fontdict=font2)
        if i == 3:    # 添加图例
            plt.legend(labels=labels,
                        prop=font1,  # Title for the legend
                        bbox_to_anchor=(1.05, 1),
                        loc='upper left',
                        borderaxespad=0
                        )

    # fig.tight_layout()#调整整体空白
    plt.subplots_adjust(wspace=0.3, hspace=0.2)#调整子图间距,wspace:横向空白,hspace:纵向空白
    plt.show()

# 各省子图
def plotDTMProvincial():
    facetnum = 3    # 设置将要绘制的相关主题个数
    fig = plt.figure(figsize=(6, 4))
    x = np.arange(2010, 2020, dtype=int)  # x坐标年份
    markes = ['-o', '-s', '-^', '-p', '-^', '-v', '-p', '-d', '-h', '-2', '-8', '-6']   #折线图标记
    for i in range(1, facetnum+1):
        sheetname = 'Sheet' + str(i)
        tmpdf = pd.read_excel(r'D:\tmpWORKSTATION\testdtm_prov1.xlsx', sheet_name=sheetname)
        labels = list(tmpdf.keys())
        y = tmpdf.transpose().values
        ax = fig.add_subplot(facetnum, 1, i)
        for j, yval in enumerate(y.tolist()):
            plt.xlim(2009.5, 2019.5)  # 设置x的范围
            ax.plot(x, yval, markes[j])
            ax.set_xticks(range(2010, 2020, 1))  # 横坐标标签
        plt.legend(labels,
                   loc='upper left',
                   bbox_to_anchor=(1.01,1),
                   borderaxespad=0,
                   labelspacing=0.2)# vertical space between legend entries

    plt.subplots_adjust(wspace=0.3, hspace=0.3)  # 调整子图间距,wspace:横向空白,hspace:纵向空白
    fig.subplots_adjust(right=0.8)
    fig.savefig('D:/test.png', dpi=600)
    # fig.show()

if __name__ == '__main__':
    # plotDTMProvincial()
    plotDTMTotal()





