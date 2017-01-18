import csv
import os
os.chdir("D:\\Users\\krodvei\\Documents\\Code\\Python\\TxtMining\\AdvokatfirmaBrækhusDege")
import numpy as np
import pickle
import gensim
import TxtMFunctions as tmfun
import impDocFunctions as impdf
import txtML

###############################################################################
# Get Sentences
###############################################################################
# Get lover
path= "D:\\Users\\krodvei\\Documents\\Code\\Python\\TxtMining\\words_databse\\raw\\lover_csv_files"
files=os.listdir(path)
asLoverRows=[]
for file in files:
	with open(path+"\\"+file, 'rt',encoding='utf-8') as asLover: #possible not <<encoding='utf-8'>>?
		data=csv.reader(asLover)
		next(data)
		tempasLoverRows=[r for r in data]
	asLoverRows += tempasLoverRows
	
lov_sentences = []  # Initialize an empty list of sentences
for row in asLoverRows:
    lov_sentences += tmfun.intext_to_sentences(row[1],remove_stopwords_stemming=False,input_type="lov")

# Get lov documents sentences
path = 'D:\\Users\\krodvei\\Documents\\Code\\Python\\TxtMining\\AdvokatfirmaBrækhusDege\\FraFirmaet\\Arbeidsavtaler (FERDIG)\\'
files=os.listdir(path)

arbdoc_sentences=[]
for i in range(len(files)):
	if str.isnumeric(files[i][0]) or str.isalpha(files[i][0]) :
		load_arbeids_doc = impdf.document_to_text(files[i], path)
		if not (load_arbeids_doc is None):
			arbdoc_sentences += tmfun.intext_to_sentences(load_arbeids_doc,remove_stopwords_stemming=False,input_type="doc")
		if i%round(len(files)/10)==0:
			pros_done=round(i/round(len(files)/10)*10)
			if pros_done!=100:
				print(str(pros_done)+'%', end='-')
			else:
				print(str(pros_done)+'%')	

# Save/Load arbeids document sentences
path="D:\\Users\\krodvei\\Documents\\Code\\Python\\TxtMining\\words_databse\\sentences_prep\\"
#pickle.dump(arbdoc_sentences, open(path+'database_lov_sentences.p', 'wb')) 
arbdoc_sentences = pickle.load(open(path+'database_lov_sentences.p', 'rb'))
				

# Get newsarticles document sentences
path = 'D:\\Users\\krodvei\\Documents\\Code\\Python\\TxtMining\\words_databse\\raw\\norwegian_news_2014\\bokmaal\\'				
directorys=[x[0] for x in os.walk(path)]
count=0
error_count=0
error_bool=False
news_sentences=[]
for directory in directorys[1:]:
	files=os.listdir(directory)
	for file in files:
		count+=1
		try:
			xml=open(directory+'\\'+file, 'rt', encoding='utf-8').read() #possible not <<encoding='utf-8'>>?
		except:
			error_count+=1
			error_bool = True
		if not error_bool:
			news_sentences += tmfun.intext_to_sentences(xml,remove_stopwords_stemming=False,input_type="xml")
		else:
			error_bool=False
		if count%1870==0:
			if count%18700==0:
				print(str(round(count/1870))+'%', end='-')
			else:
				print('', end='-')
	
# Save/Load news
path="D:\\Users\\krodvei\\Documents\\Code\\Python\\TxtMining\\words_databse\\sentences_prep\\"
#pickle.dump(news_sentences, open(path+'database_news_sentences.p', 'wb'))
news_sentences = pickle.load(open(path+'database_news_sentences.p', 'rb'))

	
			
# Make sentences
sentences=lov_sentences+arbdoc_sentences+news_sentences
path="D:\\Users\\krodvei\\Documents\\Code\\Python\\TxtMining\\words_databse\\sentences_prep\\"
#pickle.dump(sentences, open(path+'database_sentences.p', 'wb'))
sentences = pickle.load(open(path+'database_sentences.p', 'rb'))



###############################################################################
# bag of words
###############################################################################
test=[item for sublist in sentences for item in sublist]
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word",   \
					tokenizer = None,    \
					preprocessor = None, \
					stop_words = None,   \
					max_features = 5000) 
train_data_features = vectorizer.fit_transform(test)
train_data_features = train_data_features.toarray()
vocab = vectorizer.get_feature_names()
dist = np.sum(train_data_features, axis=0)

index_most_common=[i for i in range(len(dist)) if dist[i]==max(dist)]
vocab[index_most_common]



###############################################################################
# Create Word2Vec Model
###############################################################################
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

num_features = 300    # Word vector dimensionality                      
min_word_count = 10   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 20          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

from gensim.models import word2vec
print("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)
# KEEP TRAINING ON THE NEW DATA: see https://github.com/RaRe-Technologies/gensim/issues/368


model_name = "newdoclovmodelWord2Vec"
#model.save(model_name)
model = gensim.models.Word2Vec.load(model_name)

# test
model.doesnt_match('lov rett arbeidslov pensjon'.split())
model.most_similar("pensjon")
model.most_similar(positive=['kvinne', 'konge'], negative=['mann'], topn=10)
#from nltk.stem.snowball import SnowballStemmer
#stemmer = SnowballStemmer("norwegian")
#model.most_similar(stemmer.stem("arbeidsloven"))


###############################################################################
# Creat TF_IDF
###############################################################################
from sklearn.feature_extraction.text import TfidfVectorizer

# corpus should be the ML training set (here the contracts)
corpus = [' '.join(sentence) for sentence in arbdoc_sentences]
#corpus = ["This is very strange",
#          "This is very nice"]
vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus)
idf = vectorizer.idf_
tf_idf=dict(zip(vectorizer.get_feature_names(), idf))
# Full sentence set
path="D:\\Users\\krodvei\\Documents\\Code\\Python\\TxtMining\\words_databse\\sentences_prep\\"
#pickle.dump(sentences, open(path+'dict_tf_idf_fullsentences.p', 'wb'))
sentences = pickle.load(open(path+'dict_tf_idf_fullsentences.p', 'rb'))
# arbdoc_sentence set
path="D:\\Users\\krodvei\\Documents\\Code\\Python\\TxtMining\\words_databse\\sentences_prep\\"
#pickle.dump(sentences, open(path+'dict_tf_idf_arbdocsentences.p', 'wb'))
sentences = pickle.load(open(path+'dict_tf_idf_arbdocsentences.p', 'rb'))

###############################################################################
# get sentence vector
###############################################################################
arbdoc_sentences[10]
	
sentence_vec=txtML.sentence2vec(arbdoc_sentences[10],model,tf_idf)
model.similar_by_vector(sentence_vec, topn=10, restrict_vocab=None)
