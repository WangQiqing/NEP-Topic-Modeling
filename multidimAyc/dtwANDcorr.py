# plot scatter of co2 intensity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib as mpl
import matplotlib.patches as mpatches

chlabs = ['节能', '减排', '碳排放', '风电', '生物质', '新能源', '新兴产业', '煤炭', '新能源汽车', '环保',
       '低碳', '风能', '太阳能', '生物质能', '可再生能源', '电动汽车']
enlabs = ['Energy saving', 'Emission reduction', 'CO2 emission', 'WTs', 'Biomass', 'New energy', 'Emerging industry',
          'Coal', 'NEVs', 'Evironment protection', 'Low carbon', 'Wind power', 'Photovoltaics', 'Biomass power',
          'Renewable energy', 'EVs']
ch_en_dic = dict(zip(chlabs, enlabs))
provdic = {'内蒙古':'Inner Mongolia', '四川':'Sichuan', '山东':'Shandong',
           '广东':'Guangdong', '新疆':'Sinkiang', '江苏':'Jiangsu'}
pvns = [val for _,val in provdic.items()]
en_provns = list(provdic.values())
annotations = ['c-','c+','d-','d+']

dfls, indis = [], []
for i in range(6):
    df = pd.read_excel(r'F:\论文\图表\co2intensity_vs_topics.xlsx',sheet_name=en_provns[i])
    indis.append(df['Intensity'])
    cols = df.columns[2:]
    annot = annotations.copy()
    repeatls = []
    for i in range(4):
        for j in range(i+1, 4):
            if cols[i] == cols[j].split('.')[0]:
                annot.remove(annot[j])
                repeatls.append((annotations[i], annotations[j]))
    templs, newcols = [],[]
    rephead = [t[0] for t in repeatls]
    reptail = [t[1] for t in repeatls]
    for it in annot:
        i = annotations.index(it)
        if it not in rephead:
            colname = ch_en_dic[cols[i]]+'*'+it
            newcols.append(colname)
            templs.append(df[cols[i]])
        else:
            ind = annotations.index(it)
            colname = ch_en_dic[cols[ind]]+'*'+'('+it+','+reptail[rephead.index(it)]+')'
            newcols.append(colname)
            templs.append(df[cols[i]])
    tmpdf = pd.concat(templs, axis=1)
    tmpdf.columns = newcols
    dfls.append(tmpdf)

colors = ['red', 'gold', 'green','blue']
markers = ['s', '^', 'o', 'D']
xlabs = ['a)','b)','c)','d)','e)','f)']
font1={'family':'Times New Roman',
        'weight':'bold',
      'size':12}
font2={'family':'Times New Roman',
      'size':10}
fig, axes = plt.subplots(3, 2, figsize=(9,8.5))
for i, ax in enumerate(axes.flatten().tolist()):
    x = list(range(2010,2020))
    y_bar = indis[i].values
    y_lines = dfls[i].values
    ynames = [c.split('*')[0] for c in dfls[i].columns]
    line_tags = [c.split('*')[1] for c in dfls[i].columns]
    ax.bar(x, y_bar, color='tan', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=30, fontdict=font2)
    ax.set_xlabel(xlabs[i], fontdict=font1)

    yticks = [round(yt,2) for yt in np.linspace(min(y_bar)-0.2, max(y_bar)+0.2, 5)]
    ax.set_ylim(min(y_bar)-0.2, max(y_bar)+0.2)
    ax.set_yticks(yticks)
    if i in [0,2,4]:
        ax.set_ylabel('Intensity', fontdict=font1)
    ax.set_yticklabels(yticks, fontdict=font2)
    ax1 = ax.twinx()
    linenums = y_lines.shape[1]
    for j in range(linenums):
        line, = ax1.plot(x, y_lines[:,j], marker=markers[j])
        line.set_label(ynames[j])
        ax1.annotate(line_tags[j], xy=(2011,y_lines[1,j]))
        ax1.legend(fontsize=8, ncol=2, loc='lower center', framealpha=0.5, markerscale=0.7,
                   prop={'family':'Times New Roman', 'size':10}, columnspacing=0.1)
    yts = [round(yt, 2) for yt in np.linspace(np.min(y_lines) - 0.05, np.max(y_lines) + 0.015, 5)]
    ax1.set_ylim(np.min(y_lines) - 0.05, np.max(y_lines))
    ax1.set_yticks(yts)
    ax1.set_yticklabels(yts, fontdict=font2)
    if i in [1,3,5]:
        ax1.set_ylabel('Term Significance', fontdict=font1)

fig.subplots_adjust(wspace=0.28, hspace=0.32) # 调整子图间距
# fig.show()
fig.savefig('F:\\论文\\图表\\图\\多维分析\\co2DTW_PEARSON_600.png', dpi=600)

fig.savefig('F:\\论文\\图表\\图\\多维分析\\co2DTW_PEARSON_300.png', dpi=300)

#===============================================NewEnergy
# plot scatter of co2 intensity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib as mpl
import matplotlib.patches as mpatches

chlabs = ['节能', '减排', '碳排放', '风电', '生物质', '新能源', '新兴产业', '煤炭', '新能源汽车', '环保',
       '低碳', '风能', '太阳能', '生物质能', '可再生能源', '电动汽车']
enlabs = ['Energy saving', 'Emission reduction', 'CO2 emission', 'WTs', 'Biomass', 'New energy', 'Emerging industry',
          'Coal', 'NEVs', 'Evironment protection', 'Low carbon', 'Wind power', 'Photovoltaics', 'Biomass power',
          'Renewable energy', 'EVs']
ch_en_dic = dict(zip(chlabs, enlabs))
provdic = {'内蒙古':'Inner Mongolia', '四川':'Sichuan', '山东':'Shandong',
           '广东':'Guangdong', '新疆':'Sinkiang', '江苏':'Jiangsu'}
pvns = [val for _,val in provdic.items()]
en_provns = list(provdic.values())
annotations = ['c-','c+','d-','d+']

dfls, indis = [], []
for i in range(6):
    df = pd.read_excel(r'D:\1研究僧\1研究生论文\0.3新能源政策挖掘\论文\图表\NeweCap_vs_topics.xlsx',sheet_name=en_provns[i])
    indis.append(df[['Newe','pv', 'wt']])
    df = df.iloc[:, :-2]
    cols = df.columns[2:]
    annot = annotations.copy()
    repeatls = []
    for i in range(4):
        for j in range(i+1, 4):
            if cols[i] == cols[j].split('.')[0]:
                annot.remove(annot[j])
                repeatls.append((annotations[i], annotations[j]))
    templs, newcols = [],[]
    rephead = [t[0] for t in repeatls]
    reptail = [t[1] for t in repeatls]
    for it in annot:
        i = annotations.index(it)
        if it not in rephead:
            colname = ch_en_dic[cols[i]]+'*'+it
            newcols.append(colname)
            templs.append(df[cols[i]])
        else:
            ind = annotations.index(it)
            colname = ch_en_dic[cols[ind]]+'*'+'('+it+','+reptail[rephead.index(it)]+')'
            newcols.append(colname)
            templs.append(df[cols[i]])
    tmpdf = pd.concat(templs, axis=1)
    tmpdf.columns = newcols
    dfls.append(tmpdf)

colors = ['red', 'gold', 'green','blue']
markers = ['s', '^', 'o', 'D']
xlabs = ['a)','b)','c)','d)','e)','f)']
font1={'family':'Times New Roman',
        'weight':'bold',
      'size':12}
font2={'family':'Times New Roman',
      'size':10}
barcolor_list = ["darkorange", "limegreen"]
fig, axes = plt.subplots(3, 2, figsize=(9,8.5))
for i, ax in enumerate(axes.flatten().tolist()):
    x = list(range(2010,2020))
    y_bar = indis[i][["pv","wt"]].transpose().values     # 2*10
    y_lines = dfls[i].values
    ynames = [c.split('*')[0] for c in dfls[i].columns]
    line_tags = [c.split('*')[1] for c in dfls[i].columns]
    # ax.bar(x, y_bar, color='tan', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=30, fontdict=font2)
    ax.set_xlabel(xlabs[i], fontdict=font1)
    for tt in range(y_bar.shape[0]):  # i表示list的索引值
        ax.bar(x, y_bar[tt],
                width=0.5,
                bottom=np.sum(y_bar[:tt], axis=0),
                color=barcolor_list[tt % len(barcolor_list)],
                alpha=0.5)
    y_scale = indis[i]["Newe"].values
    yticks = [round(yt,2) for yt in np.linspace(min(y_scale)-0.2, max(y_scale)+0.2, 5)]
    ax.set_ylim(min(y_scale)-0.2, max(y_scale)+0.2)
    ax.set_yticks(yticks)
    if i in [0,2,4]:
        ax.set_ylabel('NEC', fontdict=font1)
    ax.set_yticklabels(yticks, rotation=30, fontdict=font2)
    ax1 = ax.twinx()
    linenums = y_lines.shape[1]
    for j in range(linenums):
        line, = ax1.plot(x, y_lines[:,j], marker=markers[j])
        line.set_label(ynames[j])
        ax1.annotate(line_tags[j], xy=(2011,y_lines[1,j]))
        ax1.legend(fontsize=8, ncol=2, loc='lower center', framealpha=0.5, markerscale=0.7,
                   prop={'family':'Times New Roman', 'size':10}, columnspacing=0.1)
    yts = [round(yt, 2) for yt in np.linspace(np.min(y_lines) - 0.05, np.max(y_lines) + 0.015, 5)]
    ax1.set_ylim(np.min(y_lines) - 0.05, np.max(y_lines))
    ax1.set_yticks(yts)
    ax1.set_yticklabels(yts, fontdict=font2)
    if i in [1,3,5]:
        ax1.set_ylabel('Term Significance', fontdict=font1)

labels = ['PVs capacity', 'WTs capacity']  # legend标签列表，上面的color即是颜色列表
patches = [mpatches.Patch(color=barcolor_list[i], label="{:s}".format(labels[i])) for i in range(len(barcolor_list))]
fig.legend(handles=patches, bbox_to_anchor = (0.65,0.93), ncol=2)
fig.subplots_adjust(wspace=0.28, hspace=0.32)  # 调整子图间距
fig.show()

# fig.savefig('F:\\论文\\图表\\图\\多维分析\\co2DTW_PEARSON_600.png', dpi=600)

fig.savefig(r'D:\1研究僧\1研究生论文\0.3新能源政策挖掘\论文\图表\图\多维分析\11NEPC11DTW_PEARSON_300.png', dpi=300)
