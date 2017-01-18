#import numpy as np
#import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords # Import the stop word list
from nltk.stem.snowball import SnowballStemmer


def text_to_wordlist(text,remove_stopwords_stemming=False):
	# fjerner alt fra "\\n0\\nEndret ved lov" og ut (info om lovendringer)
	#indexEndret = text.find("\\n0\\nEndret ved lov")

	# Remove non-letters 
	letters_only = re.sub("[^a-zæøåA-ZÆØÅ]", " ", text)
	#
	# Convert to lower case, split into individual words
	words = letters_only.lower().split()
	#
	if remove_stopwords_stemming:
		# In Python, searching a set is much faster than searching
		# a list, so convert the stop words to a set
		stops = set(stopwords.words("norwegian"))
		#
		# Remove stop words
		stemmer = SnowballStemmer("norwegian")
		words = [stemmer.stem(w) for w in words if not w in stops]
	#																					
	# Join the words back into one string separated by space, 
     # and return the result.
	return(words)# " ".join( words ))
	
###############################################################################		
#from sklearn.feature_extraction.text import CountVectorizer
#def creatBagofWords(asLoverRowsClean)
#	vectorizer = CountVectorizer(analyzer = "word",   \
#						tokenizer = None,    \
#						preprocessor = None, \
#						stop_words = None,   \
#						max_features = 5000) 
#	train_data_features = vectorizer.fit_transform(clean_train_texts)
#	train_data_features = train_data_features.toarray()
#	return(train_data_features)
###############################################################################
import nltk.data
# Load the punkt tokenizer
#tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def text_cleaner(text,input_type):
	if input_type=="lov":
		indexEndret = text.find("\\n0\\n")#Endret ved lov")
		indexDelP = text.find("\\n??Del paragraf")
		if indexEndret!=-1:
			text = text[:indexEndret]
		elif indexDelP!=-1:
			text = text[:indexDelP]
		text.replace(" jf. "," jamfør ")
		text.replace(" m.v. "," med videre ")
		text.replace(" m.m. "," med mer ")
	elif input_type=="doc":
		text = re.sub("\s[a-zA-Z]\)", " ", text)
		# punkt /r/t!!!!
		# text = re.sub("\s-\s", " ", text)
	elif input_type=="xml":
		text = re.sub("(Ã¦)|(Ã†)", "ae", text)
		text = re.sub("(Ã¸)|(Ã˜)", "oe", text)
		text = re.sub("(Ã¥)|(Ã…)", "aa", text)
		text = re.sub("(Ã©)", "e", text)
		text = re.sub("(Â[\S\s])", "", text)
		text = re.sub("(Ã[\S\s])", "", text)
		
		
	text = BeautifulSoup(text,"lxml").get_text()
	#########################################################################
	#import unicodedata
	#text=''.join((c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn'))
	#########################################################################
	#re.sub("(\d+)", " xNUMBx", test)
	text = re.sub("[æÆ]", "ae", text)
	text = re.sub("[øØ]", "oe", text)
	text = re.sub("[åÅ]", "aa", text)
	text = text.replace("\\n", ". ")
	text = re.sub("[Xx]{2,}", " xCENSOREDx ", text)
	text = re.sub("(\d+[\s.,-/:]*(\d+)*)", " xNUMBx ", text)
	text = re.sub("\(\s+xNUMBx\s+\)", " ", text)
	# improve using phone nr #text = re.sub("[\s]*(\d+[\s.,]*){2,}", " xTLFx ", text)
	# improve using dates, 
	# improve using usrl, ++
	# see regex.com
	text = re.sub("[\n\r]+", ". ", text)
	text = re.sub("[\t\f]+", " ", text)
	text = re.sub("(\s*[.,…]\s*){2,}",". ", text)
	#text = re.sub("[ ]{2,}", " ", text)
	#text = re.sub("[ ]{2,}", " ", text)
	return(text)


def intext_to_sentences(intext, remove_stopwords_stemming=False, input_type="lov" ):
	tokenizer = nltk.data.load('tokenizers/punkt/norwegian.pickle')
	# Function to split a text into parsed sentences. Returns a 
	# list of sentences, where each sentence is a list of words

	# Clean input
	intext = text_cleaner(intext,input_type)
	# 1. Use the NLTK tokenizer to split the paragraph into sentences
	raw_sentences = tokenizer.tokenize(intext.strip())
	#
	# 2. Loop over each sentence
	sentences = []
	sentences_clean=0
	for raw_sentence in raw_sentences:
		# If a sentence is empty, skip it
		if len(raw_sentence) > 0:
			# Otherwise, call text_to_wordlist to get a list of words
			sentences_clean=text_to_wordlist( raw_sentence, remove_stopwords_stemming )
			if len(sentences_clean)>3:#sentence_clean.count(' ')>3:
				if sentences_clean[0]=="xnumbx":
					is_xnumbx = True
					sent_clean_len=len(sentences_clean)
					ind_xnumbx=0
					while is_xnumbx:
						ind_xnumbx+=1
						if sentences_clean[ind_xnumbx]!="xnumbx":
							is_xnumbx=False
							sentences.append(sentences_clean[ind_xnumbx:])
						elif (sent_clean_len-1)<=ind_xnumbx:
							is_xnumbx=False
					
				else:
					sentences.append(sentences_clean)
	#
	# Return the list of sentences (each sentence is a list of words,
	# so this returns a list of lists
	return(sentences)