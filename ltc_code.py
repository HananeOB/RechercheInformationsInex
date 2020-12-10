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
from sklearn.metrics.pairwise import cosine_similarity
# from util import XMLFile, TextFile, XMLFile, Request, TextArticle
import re 
from scipy.sparse import csr_matrix


def get_all_docs(textfile="Collection_Texte" ):# Gets all documents as rows of a list
	file = open(textfile)

	text = file.read()
	text = re.sub(r'\n', ' ', text)
	# text = re.sub(r'\d+', ' ', text) # eliminer les nombres 

	patternN = r'<doc><docno>(\d+)</docno>'
	patternD = r'<doc><docno>\d+</docno>(.*?)</doc>'
	DocN = re.findall(patternN, text) ## list of DocIds 
	DocD = re.findall(patternD, text) ## list of DocTexts

	return [DocN, DocD]

def get_all_queries(queryfile="requetes.txt"):
	file = open(queryfile)
	req = file.read()
	file.close()
	patternq = r'(\d*)\s(\D*)'
	q = re.findall(patternq, req)
	return(q)


def put_query(query, docs_matrix, num_matrix): # inserts query in the last lines of the collection
	docs_matrix.append(query[1])
	num_matrix.append(query[0])
	return

def see_doc(index): 	# Visualize document of indice "index" in collection
	print("Doc number = " + num_m[index] + "\n")
	print(doc_m[index] + "\n")
	return

# Get documents and their document numbers :
num_m = get_all_docs()[0]
doc_m = get_all_docs()[1]
query_list = get_all_queries()
# Visualize first document
#see_doc(0)

# Queries :
for q in query_list :
	
	put_query(q, doc_m, num_m)
	
	# Creates vectorial matrix with corresponding weights :
	vectorizer = tf_idf(stop_words='english', sublinear_tf=True)
	matrix = vectorizer.fit_transform(doc_m)

	# Separates documents and queries
	docs = matrix[0:-1]
	query = matrix[-1]

	# ----- Evaluation function ( creer la fonction evaluation qui permet de rendre ce resultat)
	# Best fitting query result (ltc) :
	query_output = cosine_similarity(query, docs) # query_output is a list inside a list (je ne sais pas pq)
	#
	#  Sort queries :
	scores_indexed  = [(index, content) for index, content in enumerate(query_output[0]) ]
	sorted_scores  = sorted(scores_indexed , key=lambda tup: tup[1], reverse= True)[0:1500]		# Maximum results = 1500
	# ----- End of evaluation function
	
	# Writing the result to the file (the runs)
	rang = 1
	for e in sorted_scores  :
		with open("HananeJeremyWilliamYana_01_01_ltc_article_.txt", 'a') as f:
			f.write("{} Q0 {} {} {} HananeJeremyWilliamYana /article[1]\n".format(num_m[-1],num_m[e[0]],rang,e[1]))
		
		rang += 1

	## eliminer la requête de la collection
	num_m.pop() 
	doc_m.pop()
