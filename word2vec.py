import os
import pandas as pd
import nltk
import gensim
from gensim import corpora, models, similarities





corpus="Complex is better than complicated.Simple is better than complex In the face of ambiguity, refuse the temptation to guess,It seems your familiar with the Zen of Python,I am.,"
  
tok_corp= nltk.word_tokenize(corpus)
       
           
model = gensim.models.Word2Vec(tok_corp, min_count=1, size = 32)

model.save('word2vec.bin')
model = gensim.models.Word2Vec.load('word2vec.bin')
# model.most_similar('better')
# model.most_similar([vector])