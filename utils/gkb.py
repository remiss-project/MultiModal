import json
import urllib.request
import urllib.parse
import pdb
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import numpy as np


from utils.config import gkb_stop_words



def remove_punctuation(text):
    text_without_punctuation = re.sub(r'[^\w\s]', '', text)
    return text_without_punctuation




def search_gkb(query,context):
    maxa,maxdescription,maxsim=search_gkb_core(query,context)
    #if maxdescription!='':
    #   break
    ##try all subparts 
    queryall=query.split(' ')
    for queryk in queryall:
             a,description,c=search_gkb_core(queryk,context)
             if c >= maxsim:
                  maxa,maxdescription,maxsim=a,description,c  

    return maxa,maxdescription,maxsim


##########
def search_gkb_cf(query,limit):
    api_key ='AIzaSyD9n6fW29j6ThqSn1Frul56s72EONVgee0'
    service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
    params = {
    'query': query,
    'limit': limit,
    'indent': True,
    'key': api_key,
    }
    
    url      = service_url + '?' + urllib.parse.urlencode(params)
    response = json.loads(urllib.request.urlopen(url).read())
    
    
    gkb_dict={}
    for element in response['itemListElement']:
        name = element['result']['name']
        datakeys=list(element['result'].keys())

        if 'detailedDescription' in datakeys:
           description = element['result']['detailedDescription']['articleBody']
        elif 'description' in datakeys:
           description = element['result']['description']
        else:
           description=''   
        
        gkb_dict[name]=description
    

    
    return gkb_dict
    
    
    
##########
def search_gkb_topk(query, context, k):
    api_key = 'AIzaSyD9n6fW29j6ThqSn1Frul56s72EONVgee0'
    service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
    params = {
        'query': query,
        'limit': 10,
        'indent': True,
        'key': api_key,
    }

    context = remove_punctuation(context)
    url = service_url + '?' + urllib.parse.urlencode(params)
    response = json.loads(urllib.request.urlopen(url).read())
    model = SentenceTransformer('bert-base-nli-mean-tokens')

    top_k_similar = []
    
    for element in response['itemListElement']:
        name = element['result']['name']
        datakeys = list(element['result'].keys())

        if 'detailedDescription' in datakeys:
            description = element['result']['detailedDescription']['articleBody']
        elif 'description' in datakeys:
            description = element['result']['description']
        else:
            description = ''

        context_embedding = model.encode([context])[0]
        description_embedding = model.encode([description])[0]
        query_embedding = model.encode([query])[0]
        name_embedding = model.encode([name])[0]

        context_similarity = cosine_similarity([context_embedding], [description_embedding])[0][0]
        name_similarity = cosine_similarity([query_embedding], [name_embedding])[0][0]

        name_similarity = float(name_similarity > 0.5) * name_similarity
        context_similarity = float(context_similarity > 0.4)
        similarity = context_similarity*name_similarity

        top_k_similar.append((name, description, similarity))

    # Sort the top k similar elements based on their similarity scores
    top_k_similar.sort(key=lambda x: x[2], reverse=True)
    top_k_similar = top_k_similar[:k]

    return top_k_similar
def search_gkb_core(query,context):
    api_key ='AIzaSyD9n6fW29j6ThqSn1Frul56s72EONVgee0'
    service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
    params = {
    'query': query,
    'limit': 10,
    'indent': True,
    'key': api_key,
    }
    
    context = remove_punctuation(context)
    url      = service_url + '?' + urllib.parse.urlencode(params)
    response = json.loads(urllib.request.urlopen(url).read())
    
    model    = SentenceTransformer('bert-base-nli-mean-tokens')
    
    most_similar_element_description = None
    most_similar_element_name        = None
    similarity                       =-1
    max_similarity                   =-1
    

    for element in response['itemListElement']:
        name = element['result']['name']
        datakeys=list(element['result'].keys())

        if 'detailedDescription' in datakeys:
           description = element['result']['detailedDescription']['articleBody']
        elif 'description' in datakeys:
           description = element['result']['description']
        else:
           description=''   
        
        
        ####gkb filer films and books
        
        #if len(set(gkb_stop_words).intersection(description.split(' '))) >0:
        #   description='' 
        ###gkb_stop_words
        
        #element_text = f"{name} {description}"
        context_embedding = model.encode([context])[0]
        description_embedding    = model.encode([description])[0]
        query_embedding   = model.encode([query])[0]
        name_embedding    = model.encode([name])[0]
        
        # Calculate cosine similarity between the context and the current element
        context_similarity = cosine_similarity([context_embedding], [description_embedding])[0][0]
        name_similarity    = cosine_similarity([query_embedding], [name_embedding])[0][0]
        
        #old
        #name_similarity=float(name_similarity>0.4)*name_similarity
        #context_similarity=float(context_similarity>0.4)*context_similarity
        #similarity=(context_similarity + name_similarity)
        
        print('kb name: '+str(name)+ '  context_similarity: '+str(context_similarity)+ '  name_similarity: '+str(name_similarity))
       
     
        wt = (max(len(query), len(name)) / min(len(query), len(name)))  ## for same similary, if they are from different lenght strings, upvote
        wt = 1#max(1.5, wt)
        
        name_similarity    =float(name_similarity>0.4)*name_similarity*wt
        
        if len(query)>=10 and name_similarity > 0.97:
             context_similarity =float(context_similarity>0.3 ) #* (context_similarity+0.3)  
        else:
             context_similarity =float(context_similarity>0.45) #* (context_similarity+0.3)
     
        
        similarity         =context_similarity*name_similarity

        print('kb name: '+str(name)+ '  similarity: '+str(similarity)+ '  context_similarity: '+str(context_similarity)+ '  name_similarity: '+str(name_similarity))
   
            
        # Check if the current element is more similar than the previous maximum
        if similarity > max_similarity:
            max_similarity                   = similarity
            most_similar_element_name        = name
            most_similar_element_description = description
    
    #pdb.set_trace()        
    return most_similar_element_name, most_similar_element_description,max_similarity
