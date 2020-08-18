"""Visualize topic DIFF with the combination of dedrogram and heatmap"""

import numpy as np
import gensim
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from collections import OrderedDict
from gensim.matutils import jensen_shannon
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import pdist
import os
import matplotlib.colors as colors
from plotly.subplots import make_subplots

# load  LDAmodels with candidate topicnums
trained_models = OrderedDict()
topics = [28, 29, 30, 37]
for topicnum in topics:
    modelpath = os.path.join(r"D:/3policyAyc/_database/_workshop", 'Finalmodel-0603_' + str(topicnum) + '.lda')
    trained_models[topicnum] = gensim.models.LdaMulticore.load(modelpath)

print('construct figure data'+'>'*10)
dendrosFigs = []  # dendrogram data list
heatmapFigs = []  # heatmap data list
for topnum in topics:
    ldamodel = trained_models[topnum]
    # get topic distributions
    topic_dist = ldamodel.get_topics()
    # get topic terms
    num_words = 300  # based on the top 300 hundreds of words
    topic_terms = [{w for (w, _) in ldamodel.show_topic(topic, topn=num_words)} for topic in range(topic_dist.shape[0])]


    # use Jensen-Shannon distance metric in dendrogram
    def js_dist(X):
        return pdist(X, lambda u, v: jensen_shannon(u, v))


    # define method for distance calculation in clusters
    linkagefun = lambda x: sch.linkage(x, 'single')
    # Plot dendrogram
    fig = ff.create_dendrogram(topic_dist,
                               distfun=js_dist,
                               labels=list(range(1, topnum + 1)),
                               linkagefun=linkagefun,
                               color_threshold=0.3,
                               orientation='bottom')
    dendrosFigs.append(fig)
    # Create Heatmap
    mdiff, _ = ldamodel.diff(ldamodel, distance="jensen_shannon", normed=False, annotation=True)
    # get reordered topic list
    dendro_leaves = fig['layout']['xaxis']['ticktext']
    dendro_leaves = [x - 1 for x in dendro_leaves]

    # reorder distance matrix
    heat_data = mdiff[dendro_leaves, :]
    heat_data = heat_data[:, dendro_leaves]
    if topnum == 28:
        heatmapFigs.append(go.Figure(go.Heatmap(
            x=dendro_leaves,
            y=dendro_leaves,
            z=heat_data,
            colorscale='Viridis',
            colorbar = dict(title='<b>Differences</b>')
        )))
    else:
        heatmapFigs.append(go.Figure(go.Heatmap(
            x=dendro_leaves,
            y=dendro_leaves,
            z=heat_data,
            colorscale='Viridis',
            showscale=False,
            showlegend=False
        )))


# >>>>>>>>>>>>>>>>>>>>>>first subplot with a dendrogram and a heatmap<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
for i in range(len(dendrosFigs[0]['data'])):  # allocation for the 1st dendrogram
    dendrosFigs[0]['data'][i]['yaxis'] = 'y11'
    dendrosFigs[0]['data'][i]['xaxis'] = 'x11'

ticktexts = [x - 1 for x in dendrosFigs[0]['layout']['xaxis']['ticktext']]
tickvals = dendrosFigs[0]['layout']['xaxis']['tickvals']
dendrosFigs[0]['layout']['xaxis11'] = dendrosFigs[0]['layout']['xaxis']
dendrosFigs[0]['layout']['yaxis11'] = {}
dendrosFigs[0].layout.pop('xaxis')
dendrosFigs[0].layout.pop('yaxis')
dendrosFigs[0].layout.xaxis11.update({'anchor': 'y11',
                                     'domain': [0, .475],
                                     'mirror': False,
                                     'showgrid': False,
                                     'showline': False,
                                     'zeroline': False,
                                     'showticklabels': False,
                                     'ticks': ""})
dendrosFigs[0].layout.yaxis11.update({'anchor': 'x11',
                                      'domain': [.875, 1],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'showticklabels': False,
                                      'ticks': ""})
heatmapFigs[0].data[0]['x'] = tickvals
heatmapFigs[0].data[0]['y'] = tickvals
heatmapFigs[0].data[0].xaxis = 'x12'
heatmapFigs[0].data[0].yaxis = 'y12'
heatmapFigs[0]['layout']['xaxis12'] = {}
heatmapFigs[0]['layout']['yaxis12'] = {}
heatmapFigs[0].layout.xaxis12.update({'anchor': 'y12',
                                      'domain': [0, .475],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'ticktext': ticktexts,
                                     'tickvals': tickvals,
                                      'ticks': "",
                                      'tickfont':{'family':'Times New Roman','size':10},
                                      'title_text': '<b>a)</b>',
                                      'title_font': {'family':'Times New Roman','size':14},
                                      'title_standoff': 1})
heatmapFigs[0].layout.yaxis12.update({'anchor': 'x12',
                                      'domain': [.525, .9],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'ticktext': ticktexts,
                                     'tickvals': tickvals,
                                      'ticks': "",
                                      'tickfont':{'family':'Times New Roman','size':10}})

# >>>>>>>>>>>>>>>>>>>>>>2nd subplot with a dendrogram and a heatmap<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
for i in range(len(dendrosFigs[1]['data'])):  # allocation for the 2nd dendrogram
    dendrosFigs[1]['data'][i]['yaxis'] = 'y21'
    dendrosFigs[1]['data'][i]['xaxis'] = 'x21'
ticktexts = [x - 1 for x in dendrosFigs[1]['layout']['xaxis']['ticktext']]
tickvals = dendrosFigs[1]['layout']['xaxis']['tickvals']
dendrosFigs[1]['layout']['xaxis21'] = dendrosFigs[1]['layout']['xaxis']
dendrosFigs[1]['layout']['yaxis21'] = {}
dendrosFigs[1].layout.pop('xaxis')
dendrosFigs[1].layout.pop('yaxis')
dendrosFigs[1].layout.xaxis21.update({'anchor': 'y21',
                                     'domain': [.525, 1],
                                     'mirror': False,
                                     'showgrid': False,
                                     'showline': False,
                                     'zeroline': False,
                                     'showticklabels': False,
                                     'ticks': ""})
dendrosFigs[1].layout.yaxis21.update({'anchor': 'x21',
                                      'domain': [.875, 1],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'showticklabels': False,
                                      'ticks': ""})
heatmapFigs[1].data[0]['x'] = tickvals
heatmapFigs[1].data[0]['y'] = tickvals
heatmapFigs[1].data[0].xaxis = 'x22'
heatmapFigs[1].data[0].yaxis = 'y22'
heatmapFigs[1]['layout']['xaxis22'] = {}
heatmapFigs[1]['layout']['yaxis22'] = {}
heatmapFigs[1].layout.xaxis22.update({'anchor': 'y22',
                                      'domain': [.525, 1],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'ticktext': ticktexts,
                                     'tickvals': tickvals,
                                      'ticks': "",
                                      'tickfont':{'family':'Times New Roman','size':10},
                                      'title_text': '<b>b)</b>',
                                      'title_font': {'family': 'Times New Roman', 'size': 14},
                                      'title_standoff': 1})
heatmapFigs[1].layout.yaxis22.update({'anchor': 'x22',
                                      'domain': [.525, .9],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'ticktext': ticktexts,
                                     'tickvals': tickvals,
                                      'ticks': "",
                                      'tickfont':{'family':'Times New Roman','size':10}})
# >>>>>>>>>>>>>>>>>>>>>>3rd subplot with a dendrogram and a heatmap<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
for i in range(len(dendrosFigs[2]['data'])):  # allocation for the 3rd dendrogram
    dendrosFigs[2]['data'][i]['yaxis'] = 'y31'
    dendrosFigs[2]['data'][i]['xaxis'] = 'x31'
ticktexts = [x - 1 for x in dendrosFigs[2]['layout']['xaxis']['ticktext']]
tickvals = dendrosFigs[2]['layout']['xaxis']['tickvals']
dendrosFigs[2]['layout']['xaxis31'] = dendrosFigs[2]['layout']['xaxis']
dendrosFigs[2]['layout']['yaxis31'] = {}
dendrosFigs[2].layout.pop('xaxis')
dendrosFigs[2].layout.pop('yaxis')
dendrosFigs[2].layout.xaxis31.update({'anchor': 'y31',
                                     'domain': [0, .475],
                                     'mirror': False,
                                     'showgrid': False,
                                     'showline': False,
                                     'zeroline': False,
                                     'showticklabels': False,
                                     'ticks': ""})
dendrosFigs[2].layout.yaxis31.update({'anchor': 'x31',
                                      'domain': [.35, 0.475],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'showticklabels': False,
                                      'ticks': ""})
heatmapFigs[2].data[0]['x'] = tickvals
heatmapFigs[2].data[0]['y'] = tickvals
heatmapFigs[2].data[0].xaxis = 'x32'
heatmapFigs[2].data[0].yaxis = 'y32'
heatmapFigs[2]['layout']['xaxis32'] = {}
heatmapFigs[2]['layout']['yaxis32'] = {}
heatmapFigs[2].layout.xaxis32.update({'anchor': 'y32',
                                      'domain': [0, .475],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'ticktext': ticktexts,
                                     'tickvals': tickvals,
                                      'ticks': "",
                                      'tickfont':{'family':'Times New Roman','size':10},
                                      'title_text': '<b>c)</b>',
                                      'title_font': {'family': 'Times New Roman', 'size': 14},
                                      'title_standoff': 1})
heatmapFigs[2].layout.yaxis32.update({'anchor': 'x32',
                                      'domain': [0, .375],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'ticktext': ticktexts,
                                     'tickvals': tickvals,
                                      'ticks': "",
                                      'tickfont':{'family':'Times New Roman','size':10},})
# >>>>>>>>>>>>>>>>>>>>>>4th subplot with a dendrogram and a heatmap<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
for i in range(len(dendrosFigs[3]['data'])):  # allocation for the 4th dendrogram
    dendrosFigs[3]['data'][i]['yaxis'] = 'y41'
    dendrosFigs[3]['data'][i]['xaxis'] = 'x41'
ticktexts = [x - 1 for x in dendrosFigs[3]['layout']['xaxis']['ticktext']]
tickvals = dendrosFigs[3]['layout']['xaxis']['tickvals']
dendrosFigs[3]['layout']['xaxis41'] = dendrosFigs[3]['layout']['xaxis']
dendrosFigs[3]['layout']['yaxis41'] = {}
dendrosFigs[3].layout.pop('xaxis')
dendrosFigs[3].layout.pop('yaxis')
dendrosFigs[3].layout.xaxis41.update({'anchor': 'y41',
                                     'domain': [.525, 1],
                                     'mirror': False,
                                     'showgrid': False,
                                     'showline': False,
                                     'zeroline': False,
                                    'showticklabels': False,
                                     'ticks': ""})
dendrosFigs[3].layout.yaxis41.update({'anchor': 'x41',
                                      'domain': [.35, 0.475],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                     'showticklabels': False,
                                      'ticks': ""})
heatmapFigs[3].data[0]['x'] = tickvals
heatmapFigs[3].data[0]['y'] = tickvals
heatmapFigs[3].data[0].xaxis = 'x42'
heatmapFigs[3].data[0].yaxis = 'y42'
heatmapFigs[3]['layout']['xaxis42'] = {}
heatmapFigs[3]['layout']['yaxis42'] = {}
heatmapFigs[3].layout.xaxis42.update({'anchor': 'y42',
                                      'domain': [.525, 1],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'ticktext': ticktexts,
                                     'tickvals': tickvals,
                                      'ticks': "",
                                      'tickfont':{'family':'Times New Roman','size':10},
                                      'title_text': '<b>d)</b>',
                                      'title_font': {'family': 'Times New Roman', 'size': 14},
                                      'title_standoff': 1})
heatmapFigs[3].layout.yaxis42.update({'anchor': 'x42',
                                      'domain': [0, .375],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'ticktext': ticktexts,
                                     'tickvals': tickvals,
                                      'ticks': "",
                                      'tickfont':{'family':'Times New Roman','size':10}})
# heatmapFigs[3].layout.update({'title':'this is the third one'})

finalFig = go.Figure()
for i,dendro in enumerate(dendrosFigs):
    finalFig.add_traces(dendro.data)
    finalFig.layout.update(dendro.layout)
for heatmap in heatmapFigs:
    finalFig.add_trace(heatmap.data[0])
    finalFig.layout.update(heatmap.layout)

finalFig.layout.update({'width': 800, 'height': 900,
                        'showlegend': False, 'hovermode': 'closest',
                        'font':{'family':'Times New Roman','size':12,'color':'#000000'}
                        })
import plotly.io as pyio
#D:/2020止水禪心/0.3新能源政策挖掘/论文/图表/图/discuss/
pyio.write_image(finalFig, file='DIFF5.png', format='png', scale=3)