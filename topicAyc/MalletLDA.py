from __future__ import print_function
import pandas as pd
from gensim.corpora import Dictionary
import os
import gensim
import random
import pprint
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
from collections import OrderedDict
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import logging
from collections import OrderedDict

from gensim.corpora import TextCorpus, MmCorpus
from gensim import utils, models
from topicAyc.Fun_staticLDA import Fun_staticLDA

logging.basicConfig(level=logging.ERROR)  # disable warning logging


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.models import CoherenceModel


datapath = 'D:/3policyAyc/_database/_policytxt/Wordlist_all5.csv'
staticLDA = Fun_staticLDA(datapath, nobelow=5, noabove=0.7)  # initialize static LDA model
# models_dir = r"D:\3policyAyc\_database\_workshop"
# def load_models():
#     trained_models = OrderedDict()
#     for num_topics in range(40, 21, 2):
#         model_path = os.path.join(models_dir, 'lda-Malletall0530-k%d.lda' % num_topics)
#         print("Loading LDA(k=%d) from %s" % (num_topics, model_path))
#         trained_models[num_topics] = models.LdaMulticore.load(model_path)
#     return trained_models
#
# trained_models = load_models()
# topicnums,coherence_values = [],[]
# for topnum, model in trained_models.items():
#     coherencemodel = CoherenceModel(model=model, texts=staticLDA.docs, dictionary=staticLDA.dictionary, coherence='c_v')
#     coherence_values.append(coherencemodel.get_coherence())
#     topicnums.append(topnum)
# df = pd.DataFrame(zip(topicnums, coherence_values))
# df.to_excel('./coherencevary0530_part2.xlsx')
#
# mallet_path = 'C:/mallet/bin/mallet' # update this path
# ldamallet = gensim.models.wrappers.LdaMallet(mallet_path,
#                                              corpus=staticLDA.corpus,
#                                              num_topics=20,
#                                              id2word=staticLDA.dictionary,
#                                              workers=4,
#                                              alpha=50,
#                                              iterations=100)
# # Show Topics
# pprint(ldamallet.show_topics(formatted=False))
#
# # Compute Coherence Score
# coherence_model_ldamallet = CoherenceModel(model=ldamallet,
#                                            texts=staticLDA.docs,
#                                            dictionary=staticLDA.dictionary,
#                                            coherence='c_v')
# coherence_ldamallet = coherence_model_ldamallet.get_coherence()
# print('\nCoherence Score: ', coherence_ldamallet)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
limit=100; start=81; step=2;

model_dict, coherence_values = staticLDA.compute_coherence_values(limit=limit, start=start, step=step)

models_dir = r"D:\3policyAyc\_database\_workshop"
for num_topics, model in model_dict.items():
    model_path = os.path.join(models_dir, 'lda-Malletall-0530'+str(num_topics)+'.lda')
    model.save(model_path)


x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
