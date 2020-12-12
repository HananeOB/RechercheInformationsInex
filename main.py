# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:10:56 2020

@author: 	Hanane OBLOUHOU
			Jérémy LEMÉE
			William LIM
			Yana SOARES DE PAULA
"""

from numpy import argmax
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer as tf_idf
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
# from util import XMLFile, TextFile, XMLFile, Request, TextArticle
import re 
from scipy.sparse import csr_matrix

import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords

ps = PorterStemmer() 
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))  


def lemmetizing(doc):
	word_list = nltk.word_tokenize(doc)
	word_list = [w for w in word_list if not w.lower() in stop_words]  
	return( ' '.join([lemmatizer.lemmatize(w) for w in word_list]))

def stemming(doc) :
	word_list = nltk.word_tokenize(doc)
	word_list = [w for w in word_list if not w.lower() in stop_words]  
	return( ' '.join([ps.stem(w) for w in word_list]))


def get_all_docs(textfile="Collection_Texte" ):# Gets all documents as rows of a list
	file = open(textfile)

	text = file.read()
	text = re.sub(r'\n', ' ', text)
	
	patternN = r'<doc><docno>(\d+)</docno>'
	patternD = r'<doc><docno>\d+</docno>(.*?)</doc>'
	DocN = re.findall(patternN, text) ## list of DocIds 
	DocD = re.findall(patternD, text) ## list of DocTexts

	return [DocN, DocD]

# La variable Bol permet de choisir entre Lemmatisation et stemming !! 
def pretraitement(DocD, Bol=True):
	# eliminer les nombres
	DocD = [re.sub(r'\d+', ' ', doc) for doc in DocD]
	# eliminer les espaces
	DocD = [re.sub(r'\s+', ' ', doc) for doc in DocD]
	if Bol :
		# eliminer les stop words + lemmatiser --> ca prend bcp de temps troop trop long
		DocD = [lemmetizing(doc) for doc in DocD]
	else :
		# eliminer les stop words + stemming --> ca prend moins de temps  troop long
		DocD = [stemming(doc) for doc in DocD]
	return(DocD)


def get_all_queries(queryfile="requetes.txt"):
	file = open(queryfile)
	req = file.read()
	file.close()
	req = re.sub(r'\+', ' ', req)
	patternq = r'(\d*)\s(\D*)'
	q = re.findall(patternq, req)
	q = [ (num,lemmetizing(req)) for (num,req) in q]
	return(q)


def put_query(query, docs_matrix, num_matrix): # inserts query in the last lines of the collection
	docs_matrix.append(query[1])
	num_matrix.append(query[0])
	return

def see_doc(index): 	# Visualize document of indice "index" in collection
	print("Doc number = " + num_m[index] + "\n")
	print(doc_m[index] + "\n")
	return

def evaluation_ltn(docs, query) :

	# vector of scores (document and query using ltn)
	query_output = linear_kernel(query, docs, dense_output=False).toarray()[0]

	# Sorting the scores :
	scores_indexed  = [(index, content) for index, content in enumerate(query_output) ]
	sorted_scores  = sorted(scores_indexed , key=lambda tup: tup[1], reverse= True)[0:1500]		# Maximum results = 1500

	return(sorted_scores)

def evaluation_ltc(docs, query) :

	# Best fitting query result (ltc) :
	query_output = cosine_similarity(query, docs) # query_output is a list inside a list (je ne sais pas pq)
	
	# Sorting the scores :
	scores_indexed  = [(index, content) for index, content in enumerate(query_output[0]) ]
	sorted_scores  = sorted(scores_indexed , key=lambda tup: tup[1], reverse= True)[0:1500]		# Maximum results = 1500

	return(sorted_scores)

# Get documents and their document numbers :
all_docs = get_all_docs()
num_m,doc_m = all_docs[0],all_docs[1]

# Pretraitement 
doc_m = pretraitement(doc_m, Bol=False)

# # Visualize first document
# see_doc(0)

# Get queries 
query_list = get_all_queries()

# Queries :
for q in query_list :
	# add query to matrix of documents
	put_query(q, doc_m, num_m)
	
	# Creates vectorial matrix with corresponding weights :
	vectorizer = tf_idf(sublinear_tf=True, norm="l1")
	matrix = vectorizer.fit_transform(doc_m)

	# Separate docs and query
	docs = matrix[0:-1]
	query = matrix[-1]
	
	sorted_scores_ltn = evaluation_ltn(docs, query)
	sorted_scores_ltc = evaluation_ltc(docs, query)
	
	# Writing the result to the file (the runs)
	rang = 1
	for e in sorted_scores_ltn  :
		# We print to the file the first 1500 values 
		with open("HananeJeremyWilliamYana_01_01_ltn_article_stemming.txt", 'a') as f:
			f.write("{} Q0 {} {} {} HananeJeremyWilliamYana /article[1]\n".format(num_m[-1],num_m[e[0]],rang,e[1]))
		rang += 1

	rang = 1
	for e in sorted_scores_ltc  :
		# We print to the file the first 1500 values 
		with open("HananeJeremyWilliamYana_01_02_ltc_article_stemming.txt", 'a') as f:
			f.write("{} Q0 {} {} {} HananeJeremyWilliamYana /article[1]\n".format(num_m[-1],num_m[e[0]],rang,e[1]))
		rang += 1

	## eliminer la requête de la collection
	num_m.pop() 
	doc_m.pop()


