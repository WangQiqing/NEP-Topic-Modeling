"""
read and wrap data in different structure for panel analysis
"""

import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from linearmodels.panel import PooledOLS    # classic model
from linearmodels.panel import RandomEffects    # RE
from linearmodels.panel import PanelOLS  # FE
import math


def check_contain_chinese(check_str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
        else:
            return False


# read panel and topic data
provdic = {'内蒙古':'Inner Mongolia', '四川':'Sichuan', '全国':'Nationwide',
           '山东':'Shandong','广东':'Guangdong', '新疆':'Sinkiang', '江苏':'Jiangsu'}
term_endic = {'节能': 'esaving', '减排': 'emrduc', '碳排放': 'cemis', '风电': 'wt', '生物质': 'bio', '新能源': 'newe',
 '新兴产业': 'emergind', '煤炭': 'coal', '新能源汽车': 'nev', '环保': 'envrprot', '低碳': 'lcarbon', '风能': 'wenergy',
 '太阳能': 'pve', '生物质能': 'bioe', '可再生能源': 'rnewe', '装备': 'equip', '电动汽车': 'ecar'}
root_topic = 'D:\\1研究僧\\1研究生论文\\0.3新能源政策挖掘\\论文\\图表\\'
# read topic term data
topic_part1 = pd.read_excel(os.path.join(root_topic, 'term_variations_part1.xlsx'), index_col=0)
topic_part2 = pd.read_excel(os.path.join(root_topic, 'term_variations_part2.xlsx'), index_col=0)
topic_part3 = pd.read_excel(os.path.join(root_topic, 'term_variations_part3.xlsx'), index_col=0)
topic_part4 = pd.read_excel(os.path.join(root_topic, 'term_variations_part4.xlsx'), index_col=0)
# read co2 emission panel data
panel_co2 = pd.read_excel(os.path.join(root_topic, 'co2.xlsx'), index_col=0)
panel_co2.columns = [provdic[col] for col in panel_co2.columns if check_contain_chinese(col)]
panel_co2 = panel_co2.stack().reset_index()
panel_co2.columns = ['Years', 'Provs', 'CDE']
# read gdp panel data
panel_gdp = pd.read_excel(os.path.join(root_topic,'gdp.xlsx'), index_col=0)
panel_gdp.columns = [provdic[col] for col in panel_gdp.columns if check_contain_chinese(col)]
panel_gdp = panel_gdp.stack().reset_index()
panel_gdp.columns = ['Years', 'Provs', 'GDP']
# co2 intensity, 
co2_intens = panel_co2['CDE']/panel_gdp['GDP']*10000
panel_co2int = panel_co2
panel_co2int['CDE'] = co2_intens
panel_co2int.columns = list(panel_co2int.columns)[:-1]+['Intensity']


# read WT capacity panel data
panel_wt = pd.read_excel(os.path.join(root_topic,'WT.xlsx'), index_col=0)
panel_wt.columns = [provdic[col] for col in panel_wt.columns if check_contain_chinese(col)]
panel_wt = panel_wt.stack().reset_index()
panel_wt.columns = ['Years', 'Provs', 'WTs']
# read PV capacity panel data
panel_pv = pd.read_excel(os.path.join(root_topic,'PV.xlsx'), index_col=0)
panel_pv.columns = [provdic[col] for col in panel_pv.columns if check_contain_chinese(col)]
panel_pv = panel_pv.stack().reset_index()
panel_pv.columns = ['Years', 'Provs', 'PVs']
# new energy performance
newe = panel_pv['PVs']+panel_wt['WTs']
panel_newe = panel_pv.copy()
panel_newe['PVs'] = newe
panel_newe.columns = list(panel_newe.columns)[:-1]+['Newe']

# draw co2 intensity and new energy generations
all_intensity = panel_co2int.pivot_table(index=['Provs','Years'],values=['Intensity'])
all_newenergy = panel_newe.pivot_table(index=['Provs','Years'],values=['Newe'])

all_intensity.unstack().to_excel(r'D:\1研究僧\1研究生论文\0.3新能源政策挖掘\论文\图表\INTENSITYFORALL.xlsx')
all_newenergy.unstack().to_excel(r'D:\1研究僧\1研究生论文\0.3新能源政策挖掘\论文\图表\NEWENERGYFORALL.xlsx')

# plot scatter of co2 intensity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib as mpl
import matplotlib.patches as mpatches

df4_co2 = pd.merge(topic_part4, panel_co2int, on=['Years','Provs'])
provdic = {'Inner Mongolia':0, 'Sichuan':1, 'Shandong':2,'Guangdong':3, 'Sinkiang':4, 'Jiangsu':5}
provls = [k for k,_ in provdic.items()]

df1_co2 = df4_co2.drop(df4_co2[df4_co2['Provs']=='Nationwide'].index)
x = df1_co2['Intensity'].values
c = [provdic[it] for it in df1_co2['Provs']]
ydata = df1_co2.drop(['Provs', 'Years','Intensity', '装备', '新兴产业', '煤炭','低碳','风能',
                      '电动汽车','可再生能源','生物质能'], axis=1)
# normfun = lambda x: x/sum(x)
normfun = lambda x:(x-np.min(x))/(np.max(x)-np.min(x))
ydata = ydata.apply(normfun)
chlabs = ['节能', '减排', '碳排放', '风电', '生物质', '新能源', '新兴产业', '煤炭', '新能源汽车', '环保',
       '低碳', '风能', '太阳能', '生物质能', '可再生能源', '电动汽车']
enlabs = ['Energy saving', 'Emission reduction', 'CO2 emission', 'WTs', 'Biomass', 'New energy', 'Emerging industry',
          'Coal', 'NEVs', 'Evironment protection', 'Low carbon', 'Wind power', 'Photovoltaics', 'Biomass power',
          'Renewable energy', 'EVs']
ch_en_dic = dict(zip(chlabs, enlabs))

colors = ['lightskyblue', 'lime', 'red', 'gold', 'green','black']
cmap = mpl.colors.ListedColormap(colors[::-1])
font1={'family':'Times New Roman',
        'weight':'bold',
      'size':10}

fig, axes = plt.subplots(3, 3, figsize=(8, 9))
for i,ax in enumerate(axes.flatten().tolist()):
    y = ydata.iloc[:, i].tolist()
    # plt.xlim(0,np.max(x))
    ax.set_ylim(-0.1, 1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(ch_en_dic[ydata.columns[i]], fontdict=font1, rotation=90)
    ax.set_xlabel(str(i+1)+')',fontdict=font1)
    ax.scatter(x, y, c=c, s=12, cmap=cmap)

patches = [mpatches.Patch(color=colors[i], label="{:s}".format(provls[i])) for i in range(len(colors))]
fig.legend(handles=patches, ncol=6, handletextpad=0.5, columnspacing=0.5,
           prop={'family':'Times New Roman', 'weight':'bold', 'size':9}, loc='lower center')
fig.subplots_adjust(wspace=0.2, hspace=0.2) # 调整子图间距
fig.savefig(r'D:\1研究僧\1研究生论文\0.3新能源政策挖掘\论文\图表\图\多维分析\co2in.png',dpi=600)
fig.show()

# """No1.models-pooled OS"""
#
# ## co2 intensity modeling
# df1_co2 = pd.merge(topic_part1, panel_co2int, on=['Years','Provs'])
# df4_co2 = pd.merge(topic_part4, panel_co2int, on=['Years','Provs'])
#
# year = pd.Categorical(df1_co2['Years'])
# data = df1_co2.set_index(['Provs', 'Years'])
# data['Years'] = year
# data1 = data.loc[['Jiangsu','Guangdong','Shandong']]
#
# exog_vars = ["节能", "减排", "碳排放", "风电", "生物质", "新能源", "新兴产业", "煤炭", "新能源汽车", "环保", 'Years']
# exog = sm.add_constant(data1[exog_vars])
# mod = PooledOLS(data1['Intensity'], exog)
# pooled_res = mod.fit()
# print(pooled_res)
#
# # testdf = data.loc[['Jiangsu']]
# # exog_vars = ["节能", "减排", "碳排放", "风电", "生物质", "新能源", "新兴产业", "煤炭", "新能源汽车", "环保", 'Years']
# # exog = sm.add_constant(data[exog_vars])
# # mod = PooledOLS(data['Intensity'], exog)
# # pooled_res = mod.fit()
#
# """No.2 models-random effect"""
#
# mod = RandomEffects(data['Intensity'], exog)
# re_res = mod.fit()
# print(re_res)
#
# """No.3 models-fixed effect"""
# # exog_vars = ['expersq', 'union', 'married', 'year']
# # exog = sm.add_constant(data[exog_vars])
# mod = PanelOLS(data['Intensity'], exog, entity_effects=True)
# fe_res = mod.fit()
# print(fe_res)
'sort by DTM and try to fit them, group by provs'
funn1 = lambda x:(x-np.min(x))/(np.max(x)-np.min(x))
funn2 = lambda x:x/np.sum(x)
dfspecific = df4_co2.drop(df4_co2[df4_co2['Provs']=='Nationwide'].index)
ydata = dfspecific.drop(['Years', '装备', '新兴产业', '煤炭','低碳','风能',
                      '电动汽车','可再生能源','生物质能'], axis=1)
from dtw import *
def dtwDistance(array1, array2):
    alignment = dtw(array1, array2, keep_internals=True)
    return alignment.distance

dtwDis, corrls, bests = [], [], []
for prov,_ in provdic.items():
    df_intens = ydata[ydata['Provs']==prov]
    x = funn2(df_intens['Intensity'])
    print('{}:{}'.format(prov,x))
    d_from_intensity,cor,per = [], [],[]
    for i in range(9):
        coldata1 = funn2(df_intens.iloc[:,i])
        d_from_intensity.append(dtwDistance(coldata1, x))
        cor.append(coldata1.corr(pd.Series(x)))
    dtwDis.append(d_from_intensity)
    corrls.append(cor)
    besti = 0
    for i, d in enumerate(d_from_intensity):
        if d < d_from_intensity[besti]:
            besti = i
    bests.append(besti)
dtwDis_array = np.array(dtwDis)
provs = [k for k,v in provdic.items()]
df1 = pd.DataFrame(dtwDis_array.transpose(), columns=provs)
df1.to_excel(r'D:\1研究僧\1研究生论文\0.3新能源政策挖掘\论文\图表\DTWresults.xlsx')
corr_array = np.array(corrls)
df2 = pd.DataFrame(corr_array.transpose(), columns=provs)
df2.to_excel(r'D:\1研究僧\1研究生论文\0.3新能源政策挖掘\论文\图表\CORresults.xlsx')
corrInd = np.argsort(corr_array, axis=1)   # from min to max
dtwInd = np.argsort(dtwDis_array, axis=1)   # from min to max
writer = pd.ExcelWriter(r'D:\1研究僧\1研究生论文\0.3新能源政策挖掘\论文\图表\co2intensity_vs_topics.xlsx')
for i,pro in enumerate(provs):   # for each prov, plot co2intensity v.s. topics with max/min dtwDis and corrcoef
    px = list(range(10))
    # corr min and max
    corr_min = corrInd[i,:][0]
    corr_max = corrInd[i, :][-1]
    # dtw min and max
    dtw_min = dtwInd[i, :][0]
    dtw_max = dtwInd[i, :][-1]
    tempdf = ydata[ydata['Provs']==pro]
    ydata_for_plot = tempdf.drop(['Provs'], axis=1)
    ydat = ydata_for_plot.iloc[:,:9].apply(funn2)
    ydat['Intensity'] = ydata_for_plot['Intensity'].values
    # ydata_for_plot = ydata_for_plot.apply(funn2)

    cols = ydat.columns
    targed_cols = ['Intensity',cols[corr_min],cols[corr_max],
                         cols[dtw_min],cols[dtw_max]]
    py = ydat[targed_cols]
    py.to_excel(writer, sheet_name=pro)
    writer.save()

    legends = ['CO2Intensity','cormin_'+cols[corr_min],'cormax_'+cols[corr_max],
               'dtwmin_'+cols[dtw_min],'dtwmax_'+cols[dtw_max]]
    plt.plot(px, py)
    plt.title('pro')
    plt.legend(legends)


## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
dtw(df_intens.iloc[:,4], df_intens['Intensity'], keep_internals=True,
    step_pattern=rabinerJuangStepPattern(6, "c")).plot(type="twoway",offset=-2)

alignment = dtw(df_intens.iloc[:,4], df_intens['Intensity'], keep_internals=True)
dtwPlotTwoWay(alignment,df_intens.iloc[:,4], df_intens['Intensity'],ylab='a')
dtwPlotAlignment(alignment)




'----------------------------------------------------------------------------------------------------------------------'
# plot scatter of new energy capacity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib as mpl
import matplotlib.patches as mpatches

df4_nenegy = pd.merge(topic_part4, panel_newe, on=['Years','Provs'])
provdic = {'Inner Mongolia':0, 'Sichuan':1, 'Shandong':2,'Guangdong':3, 'Sinkiang':4, 'Jiangsu':5}
provls = [k for k,_ in provdic.items()]

df1_newe2 = df4_nenegy.drop(df4_nenegy[df4_nenegy['Provs']=='Nationwide'].index)
x = df1_newe2['Newe'].values
c = [provdic[it] for it in df1_newe2['Provs']]
ydata = df1_newe2.drop(['Provs', 'Years','Newe', '装备', '新兴产业', '煤炭','低碳','风能',
                      '电动汽车','可再生能源','生物质能'], axis=1)
# normfun = lambda x: x/sum(x)
normfun = lambda x:(x-np.min(x))/(np.max(x)-np.min(x))
ydata = ydata.apply(normfun)
chlabs = ['节能', '减排', '碳排放', '风电', '生物质', '新能源', '新兴产业', '煤炭', '新能源汽车', '环保',
       '低碳', '风能', '太阳能', '生物质能', '可再生能源', '电动汽车']
enlabs = ['Energy saving', 'Emission reduction', 'CO2 emission', 'WTs', 'Biomass', 'New energy', 'Emerging industry',
          'Coal', 'NEVs', 'Evironment protection', 'Low carbon', 'Wind power', 'Photovoltaics', 'Biomass power',
          'Renewable energy', 'EVs']
ch_en_dic = dict(zip(chlabs, enlabs))

colors = ['lightskyblue', 'lime', 'red', 'gold', 'green','black']
cmap = mpl.colors.ListedColormap(colors[::-1])
font1={'family':'Times New Roman',
        'weight':'bold',
      'size':10}

fig, axes = plt.subplots(3, 3, figsize=(8, 8))
for i,ax in enumerate(axes.flatten().tolist()):
    y = ydata.iloc[:, i].tolist()
    # plt.xlim(0,np.max(x))
    ax.set_ylim(-0.1, 1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(ch_en_dic[ydata.columns[i]], fontdict=font1, rotation=90)
    ax.set_xlabel(str(i+1)+')',fontdict=font1)
    ax.scatter(x, y, c=c, s=12, cmap=cmap)

patches = [mpatches.Patch(color=colors[i], label="{:s}".format(provls[i])) for i in range(len(colors))]
fig.legend(handles=patches, ncol=6, handletextpad=0.5, columnspacing=0.5,
           prop={'family':'Times New Roman', 'weight':'bold', 'size':9}, loc='lower center')
fig.subplots_adjust(wspace=0.2, hspace=0.2) # 调整子图间距
fig.show()


'sort by DTM and try to fit them, group by provs'
funn1 = lambda x:(x-np.min(x))/(np.max(x)-np.min(x))
funn2 = lambda x:x/np.sum(x)
dfspecific = df4_nenegy.drop(df4_nenegy[df4_nenegy['Provs']=='Nationwide'].index)
ydata = dfspecific.drop(['Years', '装备', '新兴产业', '煤炭','低碳','风能',
                      '电动汽车','可再生能源','生物质能'], axis=1)
from dtw import *
def dtwDistance(array1, array2):
    alignment = dtw(array1, array2, keep_internals=True)
    return alignment.distance

dtwDis, corrls, bests = [], [], []
for prov,_ in provdic.items():
    df_intens = ydata[ydata['Provs']==prov]
    x = funn2(df_intens['Newe'])
    print('{}:{}'.format(prov,x))
    d_from_intensity,cor,per = [], [],[]
    for i in range(9):
        coldata1 = funn2(df_intens.iloc[:,i])
        d_from_intensity.append(dtwDistance(coldata1, x))
        cor.append(coldata1.corr(pd.Series(x)))
    dtwDis.append(d_from_intensity)
    corrls.append(cor)
    besti = 0
    for i, d in enumerate(d_from_intensity):
        if d < d_from_intensity[besti]:
            besti = i
    bests.append(besti)
dtwDis_array = np.array(dtwDis)
provs = [k for k,v in provdic.items()]
df1 = pd.DataFrame(dtwDis_array.transpose(), columns=provs)
df1.to_excel(r'D:\1研究僧\1研究生论文\0.3新能源政策挖掘\论文\图表\DTWresults1.xlsx')
corr_array = np.array(corrls)
df2 = pd.DataFrame(corr_array.transpose(), columns=provs)
df2.to_excel(r'D:\1研究僧\1研究生论文\0.3新能源政策挖掘\论文\图表\CORresults1.xlsx')
corrInd = np.argsort(corr_array, axis=1)   # from min to max
dtwInd = np.argsort(dtwDis_array, axis=1)   # from min to max
writer = pd.ExcelWriter(r'D:\1研究僧\1研究生论文\0.3新能源政策挖掘\论文\图表\NeweCap_vs_topics.xlsx')
for i, pro in enumerate(provs):   # for each prov, plot co2intensity v.s. topics with max/min dtwDis and corrcoef
    px = list(range(10))
    # corr min and max
    corr_min = corrInd[i,:][0]
    corr_max = corrInd[i, :][-1]
    # dtw min and max
    dtw_min = dtwInd[i, :][0]
    dtw_max = dtwInd[i, :][-1]
    tempdf = ydata[ydata['Provs']==pro]
    ydata_for_plot = tempdf.drop(['Provs'], axis=1)
    ydat = ydata_for_plot.iloc[:,:9].apply(funn2)
    ydat['Newe'] = ydata_for_plot['Newe'].values
    # ydata_for_plot = ydata_for_plot.apply(funn2)

    cols = ydat.columns
    targed_cols = ['Newe',cols[corr_min],cols[corr_max],
                         cols[dtw_min],cols[dtw_max]]
    py = ydat[targed_cols]
    py["pv"] = panel_pv[panel_pv["Provs"]==pro]["PVs"].to_list()
    py["wt"] = panel_wt[panel_wt["Provs"]==pro]["WTs"].to_list()
    py.to_excel(writer, sheet_name=pro)
    writer.save()

    legends = ['NEPC','cormin_'+cols[corr_min],'cormax_'+cols[corr_max],
               'dtwmin_'+cols[dtw_min],'dtwmax_'+cols[dtw_max]]
    plt.plot(px, py)
    plt.title('pro')
    plt.legend(legends)
# """No1.models-pooled OS"""
# ## new energy performance development
# df1_nenegy = pd.merge(topic_part1, panel_newe, on=['Years','Provs'])
#
# year = pd.Categorical(panel_newe['Years'])
# data = df1_nenegy.set_index(['Provs', 'Years'])
# data['Years'] = year
# # data = data.loc[['Jiangsu','Guangdong','Shandong','Sichuan','Inner Mongolia','Sinkiang']]
#
# exog_vars = ["节能", "减排", "碳排放", "风电", "生物质", "新能源", "新兴产业", "煤炭", "新能源汽车", "环保", 'Years']
# exog = sm.add_constant(data[exog_vars])
# mod = PooledOLS(data['Newe'], exog)
# pooled_res = mod.fit()
# pooled_res.params
# pooled_res.rsquared_overall
# pooled_res.pvalues
# pooled_res.f_statistic
#
# print(pooled_res)

# """No.2 models-random effect"""
# mod = RandomEffects(data['Newe'], exog)
# re_res = mod.fit()
# print(re_res)
#
#
# """No.3 models-fixed effect"""
# # exog_vars = ['expersq', 'union', 'married', 'year']
# # exog = sm.add_constant(data[exog_vars])
# mod = PanelOLS(data['Newe'], exog, entity_effects=True)
# fe_res = mod.fit()
# print(fe_res)
