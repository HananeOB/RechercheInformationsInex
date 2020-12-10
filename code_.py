# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:10:56 2020
"""

from numpy import argmax
from sklearn.feature_extraction.text import TfidfVectorizer as tf_idf
from sklearn.metrics.pairwise import cosine_similarity
# from util import XMLFile, TextFile, XMLFile, Request, TextArticle
import re 

def get_all_docs(textfile="Collection_Texte" ):# Gets all documents as rows of a list
    file = open(textfile)

    text = file.read()
    text = re.sub(r'\n', ' ', text)

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

def see_doc(index):     # Visualize document of indice "index" in collection
    print("Doc number = " + num_m[index] + "\n")
    print(doc_m[index] + "\n")
    return

# Get documents and their document numbers :
num_m = get_all_docs()[0]
doc_m = get_all_docs()[1]
query_list = get_all_queries()


# Queries :
for q in query_list :
    put_query(q, doc_m, num_m)
    
    # Creates vectorial matrix with corresponding weights :
    vectorizer = tf_idf(stop_words='english', sublinear_tf=True)
    matrix = vectorizer.fit_transform(doc_m )
    
    #print(matrix.toarray())                    # Matrix with weights
    #print(vectorizer.get_feature_names())        # Gets normalized terms of the documents

    # Separates documents and queries

    # # Best fitting query result :
    scores = cosine_similarity(matrix[0:-1], matrix[-1]) # query_output is a list inside a list (je ne sais pas pq)
    # # Sorts queries :
    scores_indexed  = [(index, content) for index, content in enumerate(scores[0])]
    C = sorted(scores_indexed, key=lambda tup: tup[1], reverse= True)[0:1500]        # Maximum results = 1500
    print(C)
    # rang = 1
    # for e in C :
        
    #     with open("HananeJeremyWilliamYana_01_01_TF-IDF_article_.txt", 'a') as f:
    #         f.write("{} Q0 {} {} {} HananeJeremyWilliamYana /article[1]\n".format(num_m[-1],num_m[e[0]],rang,e[1]))
        
    #     rang += 1
    #     ## print("content = ", doc_m[e[0]])

    ## eliminer la requête de la collection
    num_m.pop() 
    doc_m.pop()


# print("index in query_output = ", index)
# print("doc num = ", num_m[index])
# print("content = ", doc_m[index])

############### Queries ##################
# Query 1 : computer science algorithms
# Ambigue query

# Query 2 : where is Church Lawton
# Specific query

# Query 3 : nuclear plants in europe

# Query 4 : scientology definition

# Query 5 : presidents of the united states

# Query 6 : open source softwares

# Query 7 : easy recipe
######################################

# TODO: output of requests inside "run"


#xmlfile=XMLFile("Collection_XML.xml")
