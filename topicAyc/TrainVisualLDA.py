"""
Run after you have chosen an proper topic number
Training visualization of LDA
"""
from gensim.models import ldamodel
from gensim.corpora import Dictionary
import pandas as pd
import re
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation

import numpy as np

df = pd.read_excel('./_database/_policytxt/wordlist_Jiangsu.xlsx')
docs = df['ptext'].tolist()
for idx in range(len(docs)):
    docs[idx] = docs[idx].split('\n')   # Split into words.

docs = [[token for token in doc if len(token) > 1] for doc in docs]     # remove rare words
dictionary = Dictionary(docs)   # LDA dic

c1 = int(len(docs)*0.5)
c2 = c1+int(len(docs)*0.25)
training_docs = docs[:c1]
holdout_docs = docs[c1:c2]
test_docs = docs[c2:]

training_corpus = [dictionary.doc2bow(text) for text in training_docs]
holdout_corpus = [dictionary.doc2bow(text) for text in holdout_docs]
test_corpus = [dictionary.doc2bow(text) for text in test_docs]

# python -m visdom.server
from gensim.models.callbacks import CoherenceMetric, DiffMetric, PerplexityMetric, ConvergenceMetric

# define perplexity callback for hold_out and test corpus
pl_holdout = PerplexityMetric(corpus=holdout_corpus, logger="visdom", title="Perplexity (hold_out)")
pl_test = PerplexityMetric(corpus=test_corpus, logger="visdom", title="Perplexity (test)")

# define other remaining metrics available
ch_umass = CoherenceMetric(corpus=training_corpus, coherence="u_mass", logger="visdom", title="Coherence (u_mass)")
ch_cv = CoherenceMetric(corpus=training_corpus, texts=training_docs, coherence="c_v", logger="visdom", title="Coherence (c_v)")
diff_kl = DiffMetric(distance="kullback_leibler", logger="visdom", title="Diff (kullback_leibler)")
convergence_kl = ConvergenceMetric(distance="jaccard", logger="visdom", title="Convergence (jaccard)")

callbacks = [pl_holdout, pl_test, ch_umass, ch_cv, diff_kl, convergence_kl]

# training LDA model
model = ldamodel.LdaModel(
    corpus=training_corpus,
    id2word=dictionary,
    num_topics=10,
    passes=30,
    chunksize=450,
    iterations=200,
    alpha='auto',
    callbacks=callbacks)
model.save('LDA_Jiangsu')


"""=================================================================================================================="""

# to get a metric value on a trained model
print(CoherenceMetric(corpus=training_corpus, coherence="u_mass").get_value(model=model))

import logging
from gensim.models.callbacks import CoherenceMetric, DiffMetric, PerplexityMetric, ConvergenceMetric

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# define perplexity callback for hold_out and test corpus
pl_holdout = PerplexityMetric(corpus=holdout_corpus, logger="shell", title="Perplexity (hold_out)")
pl_test = PerplexityMetric(corpus=test_corpus, logger="shell", title="Perplexity (test)")

# define other remaining metrics available
ch_umass = CoherenceMetric(corpus=training_corpus, coherence="u_mass", logger="shell", title="Coherence (u_mass)")
diff_kl = DiffMetric(distance="kullback_leibler", logger="shell", title="Diff (kullback_leibler)")
convergence_jc = ConvergenceMetric(distance="jaccard", logger="shell", title="Convergence (jaccard)")

callbacks = [pl_holdout, pl_test, ch_umass, diff_kl, convergence_jc]

# training LDA model
model = ldamodel.LdaModel(corpus=training_corpus,
                          id2word=dictionary,
                          num_topics=35,
                          passes=2,
                          eval_every=None,
                          callbacks=callbacks)
model.metrics

