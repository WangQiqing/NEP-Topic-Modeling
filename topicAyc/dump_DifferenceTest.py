"""计算主题相似度，生成热力图"""

from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation

import numpy as np
import pandas as pd
import re
import gensim
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
# from plotly.subplots import make_subplots
from topicAyc.Fun_staticLDA import Fun_staticLDA
from collections import OrderedDict
import matplotlib.pyplot as plt
from gensim.matutils import jensen_shannon
from scipy import spatial as scs
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import pdist


staticLDA = Fun_staticLDA(dpath='D:/3policyAyc/_database/_policytxt/Wordlist_all5.csv', nobelow=20,noabove=0.7)
topics = [36, 38]
trained_models = OrderedDict()
for topicnum in topics:
    print("Training LDA(k=%d)" % topicnum)
    ldamodel = gensim.models.LdaMulticore(staticLDA.mmcorpus,
                                          id2word=staticLDA.dictionary,
                                          num_topics=topicnum,
                                          workers=4,
                                          passes=10,
                                          iterations=500,
                                          random_state=42,
                                          eval_every=None,
                                          alpha='asymmetric',  # shown to be better than symmetric in most cases
                                          decay=0.5, offset=64  # best params from Hoffman paper
                                          )
    trained_models[topicnum] = ldamodel

import os

models_dir = r"D:\3policyAyc\_database\_workshop"
for num_topics, model in trained_models.items():
    model_path = os.path.join(models_dir, 'Finalmodel-0603_'+str(num_topics)+'.lda')
    model.save(model_path)


    # ax.imshow(heat_data_sorted)
# # We want to show all ticks...
# ax.set_xticks(np.arange(1,topicnums+1))
# ax.set_yticks(np.sort(np.arange(1,topicnums+1)))
# # ... and label them with the respective list entries





#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
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


trained_models = OrderedDict()
topics = [28,29,30,36,37,38]
for topicnum in topics:
    modelpath = os.path.join(r"D:\3policyAyc\_database\_workshop", 'Finalmodel-0603_'+str(topicnum)+'.lda')
    trained_models[topicnum] = gensim.models.LdaMulticore.load(modelpath)
FIGs = []
for topnum in topics:
    topnum=28
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
                               labels=list(range(1, topnum+1)),
                               linkagefun=linkagefun,
                               color_threshold=0.3,
                               orientation='bottom')
    for i in range(len(fig['data'])):
        fig['data'][i]['yaxis'] = 'y2'
        fig['data'][i]['xaxis'] = 'x'

    fig1 = ff.create_dendrogram(topic_dist,
                               distfun=js_dist,
                               labels=list(range(1, topnum+1)),
                               linkagefun=linkagefun,
                               color_threshold=0.3,
                               orientation='bottom')
    for i in range(len(fig['data'])):
        fig1['data'][i]['yaxis'] = 'y2'
        fig1['data'][i]['xaxis'] = 'x2'
    for data in fig1.data:
        fig.add_trace(data)
    # fig['layout'].update({'width': 1000, 'height': 600})

    # py.plot(dendro)
    # Create Heatmap
    mdiff, _ = ldamodel.diff(ldamodel, distance="jensen_shannon", normed=False, annotation=True)
    # get reordered topic list
    dendro_leaves = fig['layout']['xaxis']['ticktext']
    dendro_leaves = [x - 1 for x in dendro_leaves]

    # reorder distance matrix
    heat_data = mdiff[dendro_leaves, :]
    heat_data = heat_data[:, dendro_leaves]
    heatmap = [
        go.Heatmap(
            x=dendro_leaves,
            y=dendro_leaves,
            z=heat_data,
            colorscale='Blues',
            xaxis='x',
            yaxis='y'
        ),
        go.Heatmap(
            x=dendro_leaves,
            y=dendro_leaves,
            z=heat_data,
            colorscale='Blues',
            xaxis='x2',
            yaxis='y'
        )
    ]
    heatmap[0]['x'] = fig['layout']['xaxis']['tickvals']
    heatmap[0]['y'] = fig['layout']['xaxis']['tickvals']
    heatmap[1]['x'] = fig['layout']['xaxis']['tickvals']
    heatmap[1]['y'] = fig['layout']['xaxis']['tickvals']
    # Add Heatmap Data to Figure
    for data in heatmap:
        fig.add_trace(data)
#####################################################################
        # Edit Layout
    fig.update_layout({'width':1000, 'height':800,
                             'showlegend':False, 'hovermode': 'closest',
                             })
    # Edit xaxis
    fig.update_layout(xaxis={'domain': [0.05, 0.4],
                              'mirror': False,
                              'showgrid': False,
                              'showline': False,
                              'showticklabels': True,
                              'tickmode': "array",
                              'ticktext': dendro_leaves,
                              'tickvals': fig['layout']['xaxis']['tickvals'],
                              'zeroline': False,
                              'ticks': ""},
                      yaxis={'domain': [0, .85],
                             'mirror': False,
                             'showgrid': False,
                             'showline': False,
                             'showticklabels': True,
                             'tickmode': "array",
                             'ticktext': dendro_leaves,
                             'tickvals': fig['layout']['xaxis']['tickvals'],
                             'zeroline': False,
                             'ticks': ""
                             })

    # Edit yaxis
    # fig.update_layout(yaxis={'domain': [0, .85],
    #                           'mirror': False,
    #                           'showgrid': False,
    #                           'showline': False,
    #                           'showticklabels': True,
    #                           'tickmode': "array",
    #                           'ticktext': dendro_leaves,
    #                           'tickvals': fig['layout']['xaxis']['tickvals'],
    #                           'zeroline': False,
    #                           'ticks': ""
    #                         })
    # Edit yaxis2
    fig.update_layout(yaxis2={'domain':[.825, .975],
                                       'mirror': False,
                                       'showgrid': False,
                                       'showline': False,
                                       'zeroline': False,
                                       'showticklabels': False,
                                       'ticks':""})
    fig.update_layout(xaxis2={'domain': [.5, 1],
                              'mirror': False,
                              'showgrid': False,
                              'showline': False,
                              'showticklabels': True,
                              'tickmode': "array",
                              'ticktext': dendro_leaves,
                              'tickvals': fig['layout']['xaxis']['tickvals'],
                              'zeroline': False,
                              'ticks': ""})

    # Edit yaxis
    fig.update_layout(yaxis3={'domain': [0, .85],
                              'mirror': False,
                              'showgrid': False,
                              'showline': False,
                              'showticklabels': True,
                              'tickmode': "array",
                              'ticktext': dendro_leaves,
                              'tickvals': fig['layout']['xaxis']['tickvals'],
                              'zeroline': False,
                              'ticks': ""
                            })

    py.plot(fig)

    # Plot!
    FIGs.append(fig)
    # py.plot(fig)
FIGs[0].update_layout(showlegend=False)
py.plot(FIGs[0])





'''Plot heatmap'''
fig = plt.figure(figsize=(10,10))
for ind, topicnum in enumerate(topics):
    # get topic distributions
    topic_dist = trained_models[topicnum].get_topics()
    # get topic terms
    num_words = 300  # based on the top 300 hundreds of words
    topic_terms = [{w for (w, _) in trained_models[topicnum].show_topic(topic, topn=num_words)}
                   for topic in range(topic_dist.shape[0])]
    # use Jensen-Shannon distance metric in dendrogram
    def js_dist(X):
        return pdist(X, lambda u, v: jensen_shannon(u, v))
    # define method for distance calculation in clusters
    linkagefun = lambda x: sch.linkage(x, 'single')
    # Initialize figure by creating upper dendrogram
    figure = ff.create_dendrogram(topic_dist, distfun=js_dist, labels=list(range(1, topicnum+1)), linkagefun=linkagefun)
    # get distance matrix
    mdiff, _ = trained_models[topicnum].diff(trained_models[topicnum], distance="jensen_shannon", normed=False)
    # get reordered topic list
    dendro_leaves = figure['layout']['xaxis']['ticktext']
    dendro_leaves = [x - 1 for x in dendro_leaves]
    xylabels = ['T'+str(t+1) for t in dendro_leaves]
    # reorder distance matrix
    heat_data = mdiff[dendro_leaves, :]
    heat_data = heat_data[:, dendro_leaves]
    # plot heatmap
    topicnums = len(mdiff)
    heat_data_sorted = heat_data[np.arange(topicnums-1,-1,-1),:]
    discrete_mdiff = np.digitize(heat_data_sorted, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5])  # discretization to 6 categories
    cmap = colors.ListedColormap(['#33ff99', '#33ff33', '#99ff33', '#ffff33', '#ff9933', '#ff3333'])
    ax = fig.add_subplot(3, 2, ind+1)
    ax.set_xticks(np.arange(0,topicnums))
    ax.set_yticks(np.arange(0,topicnums))
    # labels
    ax.set_xticklabels(xylabels, rotation=90)
    xylabels.reverse()
    ax.set_yticklabels(xylabels)
    ax.imshow(discrete_mdiff, cmap=cmap)
fig.show()


# import numpy as np
# import seaborn as sb
# import matplotlib.pyplot as plt
# data = np.random.rand(4, 6)
# sb.palplot(sb.mpl_palette("Set3", 11))
# heat_map = sb.heatmap(data, cmap="cubehelix")
#
#
#
# # Plot heatmap
# annotation = text_annotation(topic_dist, topic_terms, n_ann_terms, linkagefun)
#
# # Initialize figure by creating upper dendrogram
# figure = ff.create_dendrogram(topic_dist, distfun=js_dist, labels=list(range(1, topnum+1)), linkagefun=linkagefun, hovertext=annotation)
# for i in range(len(figure['data'])):
#     figure['data'][i]['yaxis'] = 'y2'
#
# # get distance matrix and it's topic annotations
# mdiff, annotation = ldamodel.diff(ldamodel, distance="jensen_shannon", normed=False, annotation=True)
#
# # get reordered topic list
# dendro_leaves = figure['layout']['xaxis']['ticktext']
# dendro_leaves = [x - 1 for x in dendro_leaves]
#
# # reorder distance matrix
# heat_data = mdiff[dendro_leaves, :]
# heat_data = heat_data[:, dendro_leaves]
#
# # heatmap annotation
# annotation_html = [["+++ {}<br>--- {}".format(", ".join(int_tokens), ", ".join(diff_tokens))
#                     for (int_tokens, diff_tokens) in row] for row in annotation]
#
# # plot heatmap of distance matrix
# heatmap = go.Data([
#     go.Heatmap(
#         z=heat_data,
#         colorscale='YIGnBu',
#         text=annotation_html,
#         hoverinfo='x+y+z+text'
#     )
# ])
#
# heatmap[0]['x'] = figure['layout']['xaxis']['tickvals']
# heatmap[0]['y'] = figure['layout']['xaxis']['tickvals']
#
# # Add Heatmap Data to Figure
# figure['data'].extend(heatmap)
#
# dendro_leaves = [x + 1 for x in dendro_leaves]
#
# # Edit Layout
# figure['layout'].update({'width': 800, 'height': 800,
#                          'showlegend':False, 'hovermode': 'closest',
#                          })
#
# # Edit xaxis
# figure['layout']['xaxis'].update({'domain': [.25, 1],
#                                   'mirror': False,
#                                   'showgrid': False,
#                                   'showline': False,
#                                   "showticklabels": True,
#                                   "tickmode": "array",
#                                   "ticktext": dendro_leaves,
#                                   "tickvals": figure['layout']['xaxis']['tickvals'],
#                                   'zeroline': False,
#                                   'ticks': ""})
# # Edit yaxis
# figure['layout']['yaxis'].update({'domain': [0, 0.75],
#                                   'mirror': False,
#                                   'showgrid': False,
#                                   'showline': False,
#                                   "showticklabels": True,
#                                   "tickmode": "array",
#                                   "ticktext": dendro_leaves,
#                                   "tickvals": figure['layout']['xaxis']['tickvals'],
#                                   'zeroline': False,
#                                   'ticks': ""})
# # Edit yaxis2
# figure['layout'].update({'yaxis2':{'domain': [0.75, 1],
#                                    'mirror': False,
#                                    'showgrid': False,
#                                    'showline': False,
#                                    'zeroline': False,
#                                    'showticklabels': False,
#                                    'ticks': ""}})
#
# # import plotly
# # plotly.tools.set_credentials_file(username='wqq1300', api_key='wqqzy19941')
# py.plot(figure, image_filename='D:topic29')
# py.offline.plot()



