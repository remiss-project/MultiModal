#!/usr/bin/env python
# coding: utf-8

# In[27]:

import argparse
import requests
import os
import PIL
import shutil
from PIL import Image
import imghdr
from bs4 import BeautifulSoup
import bs4
import time
import pdb
from googleapiclient.discovery import build
#from google.cloud import vision
import io
import os
from bs4 import NavigableString
import json
from utils.web_scraping_utils import get_captions_from_page, save_html
import time
from utils.config import save_folder_path,split_type,sub_split,dir_ser_api,gse_cid
from urllib.parse import urlparse

#####################################################################################################


############################################################################################################
parser = argparse.ArgumentParser(description='Download dataset for direct search queries')              
parser.add_argument('--how_many_queries', type=int, default=1,
                    help='how many query to issue for each item - each query is 10 images')
parser.add_argument('--continue_download', type=int, default=0,
                    help='whether to continue download or start from 0 - should be 0 or 1')

parser.add_argument('--how_many', type=int, default=-1,
                    help='how many items to query and download, 0 means download untill the end')
parser.add_argument('--end_idx', type=int, default=-1,
                    help='where to end, if not specified, will be inferred from how_many')    
parser.add_argument('--start_idx', type=int, default=-1,
                    help='where to start, if not specified will be inferred from the current saved json or 0 otherwise')
parser.add_argument('--hashing_cutoff', type=int, default=15,
                    help='threshold used in hashing')
args = parser.parse_args()
#################################################################################333



my_api_key = dir_ser_api
my_cse_id =  gse_cid



full_save_path = os.path.join(save_folder_path,split_type,'direct_search',sub_split)
if not os.path.exists(full_save_path):
    os.makedirs(full_save_path)
    
##file for saving errors in saving..
if os.path.isfile(os.path.join(full_save_path,'unsaved.txt')) and args.continue_download:
    saved_errors_file= open(os.path.join(full_save_path,'unsaved.txt'), "a")
else:
    saved_errors_file= open(os.path.join(full_save_path,'unsaved.txt'), "w")

##file for keys with no annotations..
if os.path.isfile(os.path.join(full_save_path,'no_annotations.txt')) and args.continue_download:
    no_annotations_file= open(os.path.join(full_save_path,'no_annotations.txt'), "a")
else:
    no_annotations_file= open(os.path.join(full_save_path,'no_annotations.txt'), "w")

# json file containing the index and path of all downloaded items. 
json_download_file_name = os.path.join(full_save_path,sub_split+'.json') 

#continue using the current saved json file -- don't start a new file -- load the saved dict 
if os.path.isfile(json_download_file_name) and os.access(json_download_file_name, os.R_OK) and args.continue_download:
    with open(json_download_file_name, 'r') as fp:
        all_direct_annotations_idx = json.load(fp)
#start a new file -- start from an empty dict 
else:
    with io.open(json_download_file_name, 'w') as db_file:
        db_file.write(json.dumps({}))
    with io.open(json_download_file_name, 'r') as db_file:
        all_direct_annotations_idx = json.load(db_file)








def google_search(search_term, api_key, cse_id, how_many_queries, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res_list = []
    for i in range(0,how_many_queries):
        start = i*10 + 1
        #if remiss_bcn:
        #res = service.cse().list(q=search_term, searchType='image', num = 10, cr="countryES",lr="lang_es", start=start, cx=cse_id, **kwargs).execute()    
        #else:      
        res = service.cse().list(q=search_term, searchType='image', num = 10, start=start, cx=cse_id, **kwargs).execute()    
        res_list.append(res)
    return res_list



def download_and_save_image(image_url, save_folder_path, file_name):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        
        #oldways
        #response = requests.get(image_url,stream = True,timeout=(120,120))
        response =requests.get(image_url,stream = True,timeout=(120,120), headers=headers)
        ##
        
        if response.status_code == 200:
            response.raw.decode_content = True
            image_path = os.path.join(save_folder_path,file_name+'.jpg')
            with open(image_path,'wb') as f:
                shutil.copyfileobj(response.raw, f)
            if imghdr.what(image_path).lower() == 'png':
                img_fix = Image.open(image_path)
                img_fix.convert('RGB').save(image_path)
                print('trying to save')
                print(image_path)
            if os.path.exists(image_path):
                print('img saved')
                return 1  # Return success status code
            else:
                print('img not saved')
                return 0 
        else:
            return 0
    except:
        return 0 
        
        
        
        
        
def old_download_and_save_image(image_url, save_folder_path, file_name):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        
        #oldways
        #response = requests.get(image_url,stream = True,timeout=(120,120))
        response =requests.get(image_url,stream = True,timeout=(120,120), headers=headers)
        ##
        
        if response.status_code == 200:
            response.raw.decode_content = True
            image_path = os.path.join(save_folder_path,file_name+'.jpg')
            with open(image_path,'wb') as f:
                shutil.copyfileobj(response.raw, f)
            if imghdr.what(image_path).lower() == 'png':
                img_fix = Image.open(image_path)
                img_fix.convert('RGB').save(image_path)
            return 1 
        else:
            return 0
    except:
        return 0 

def get_direct_search_annotation(search_results_lists,save_folder_path):
    image_save_counter = 0 
    
    direct_annotation = {}
    images_with_captions = []
    images_with_captions_matched_tags = []
    images_with_no_captions = []
    
    
    #pdb.set_trace()
    #print('stop -- here')
    for one_result_list in search_results_lists:
        if 'items' in one_result_list.keys():
            items=one_result_list['items']
            print('number of results: '+str(len(items)))
            for item in items:
                image={}
                caption = {}
                if 'link' in item.keys():
                    image['img_link'] = item['link']
                if 'contextLink' in item['image'].keys():          
                    image['page_link'] = item['image']['contextLink'] 
                if 'displayLink' in item.keys():            
                    image['domain'] = item['displayLink']
                if 'snippet' in item.keys():
                    image['snippet'] = item['snippet']
                #download and save images
                print('trying to download this:'+item['link'])
                download_status = download_and_save_image(item['link'], save_folder_path, str(image_save_counter))
                #if the image cannot be downloaded, skip..
                if download_status == 0:
                    print('the image cannot be downloaded, skip,------------------- what !!')
                    #pdb.set_trace()
                    continue                
                image['image_path'] = os.path.join(save_folder_path,str(image_save_counter)+'.jpg')
            
                try:
                    caption,title,code,req = get_captions_from_page(item['link'],item['image']['contextLink'])
                except: 
                    print('Error happened in getting captions')
                    #pdb.set_trace()
                    continue 
                
                ##
                #print(caption)
                #pdb.set_trace()
                ##
                    
                saved_html_flag = save_html(req, os.path.join(save_folder_path,str(image_save_counter)+'.txt'))     
                if saved_html_flag:            
                    image['html_path'] = os.path.join(save_folder_path,str(image_save_counter)+'.txt')
                
                if len(code)>0 and (code[0] == '5' or code[0] == '4'):
                    image['is_request_error'] = True 
                if 'title' in item.keys():
                    if item['title'] is None: item['title'] = ''
                else:
                    item['title'] = ''            
                if title is None: title = ''       
                if len(title) > len( item['title'].lstrip().rstrip()):
                    image['page_title'] = title
                else:
                    image['page_title'] = item['title']

                if caption:
                    image['caption'] = caption
                    images_with_captions.append(image)
                else:
                    try:
                        caption,title,code,req = get_captions_from_page(item['link'],item['image']['contextLink'],req,args.hashing_cutoff)
                    except: 
                        print('Error happened in getting captions')
                        pdb.set_trace()
                        continue 
                    if caption: 
                        image['caption'] = caption          
                        images_with_captions_matched_tags.append(image)
                    else:
                        images_with_no_captions.append(image)
                image_save_counter = image_save_counter + 1
    
    
    if len(images_with_captions) == 0 and len(images_with_no_captions) == 0 and len(images_with_captions_matched_tags) == 0:
        direct_annotation = {}
    else:
        direct_annotation['images_with_captions'] = images_with_captions
        direct_annotation['images_with_no_captions'] = images_with_no_captions
        direct_annotation['images_with_caption_matched_tags'] = images_with_captions_matched_tags

    #pdb.set_trace()
    return direct_annotation    

def search_and_save_one_query(text_query, id_in_clip):
    new_folder_path = os.path.join(full_save_path,str(id_in_clip))
    if not os.path.exists(new_folder_path):
      os.makedirs(new_folder_path)
    result = google_search(text_query, my_api_key, my_cse_id,how_many_queries=args.how_many_queries)
    
    #pdb.set_trace()
    if int(result[0]['searchInformation']['totalResults'])==0:
       try:
          corrected_query= result[0]['spelling']['correctedQuery']
          result = google_search(corrected_query, my_api_key, my_cse_id,how_many_queries=args.how_many_queries)
          if int(result[0]['searchInformation']['totalResults'])>0:
              print('_______________ correction helps')
          else:
              print('_______________ correction does not helps')
       except:
          print('_______________  correction failed')
    
    
    print(len(result))
    print(result)

    if int(result[0]['searchInformation']['totalResults'])>0:
           direct_search_results = get_direct_search_annotation(result,new_folder_path)
           new_json_file_path   = os.path.join(new_folder_path,'direct_annotation.json')
           save_json_file(new_json_file_path, direct_search_results, id_in_clip)
           return new_json_file_path
    else:
           print('found nothing')
           return False


def search_and_save_one_query_dyn(text_query, id_in_clip,k):
    tfile=str(id_in_clip)+'-'+str(k)
    new_folder_path = os.path.join(full_save_path,tfile)
    if not os.path.exists(new_folder_path):
      os.makedirs(new_folder_path)
      
    if not os.path.exists(new_folder_path):
       print('step1 folder not created')
       pdb.set_trace()   
    result = google_search(text_query, my_api_key, my_cse_id,how_many_queries=args.how_many_queries)
    
    #pdb.set_trace()
    if int(result[0]['searchInformation']['totalResults'])==0:
       try:
          #pdb.set_trace()
          corrected_query= result[0]['spelling']['correctedQuery']
          result = google_search(corrected_query, my_api_key, my_cse_id,how_many_queries=args.how_many_queries)
          if int(result[0]['searchInformation']['totalResults'])>0:
              print('_______________ correction helps')
          else:
              print('_______________ correction does not helps')
       except:
          print('_______________  correction failed')
    
    
    print(len(result))
    print(result)
    print(new_folder_path)
    os.makedirs(new_folder_path, exist_ok=True)
    print('folder path !!!')
    ####pdb.set_trace()
    if int(result[0]['searchInformation']['totalResults'])>0:
           direct_search_results = get_direct_search_annotation(result,new_folder_path)
           new_json_file_path    = os.path.join(new_folder_path,'direct_annotation.json')
           save_json_file(new_json_file_path, direct_search_results, id_in_clip)
           return new_json_file_path
    else:
           print('found nothing')
           return False
        

def save_json_file(file_path, dict_file, cur_id_in_clip, saving_idx_file=False):
    global all_direct_annotations_idx, saved_errors_file
    #load the previous saved file 
    if saving_idx_file:
        with open(file_path, 'r') as fp:
            old_idx_file = json.load(fp)  
    try:
        with io.open(file_path, 'w') as db_file:
            json.dump(dict_file, db_file)
    except:
        saved_errors_file.write(str(cur_id_in_clip)+'\n')
        saved_errors_file.flush()
        if saving_idx_file:
            all_direct_annotations_idx = old_idx_file 
            with io.open(file_path, 'w') as db_file:
                json.dump(old_idx_file, db_file)



#def extract_domain(url):
#        parsed_url = urlparse(url)
#        return parsed_url.netloc


def getdatafromjson(jsonfile):

    print('retrieving data from direct search')
    #pdb.set_trace()
    f = open(jsonfile)
    data = json.load(f)
    data_wc =data['images_with_captions']
    #data_wnmt =data['images_with_captions_matched_tags'] dir_ser_adaptive
    data_wnc =data['images_with_no_captions']
    img_paths=[]
    img_caps=[]
    img_domains=[]
    ret_cap=''
    for datum in data_wc:
        img_paths.append(datum['image_path'])
        try:
           ret_cap=str ( datum['caption'] ) + datum['page_title']  ## editidex, bc caps are fked for remiss,always using titles
        except:
           ret_cap=''   
        img_caps.append(ret_cap)
        img_domains.append(datum['domain'])
        print(ret_cap)
        
    for datum in data_wnc:
        img_paths.append(datum['image_path'])   
        try:
           ret_cap=datum['page_title']
        except:
           ret_cap='' 
        img_caps.append(ret_cap)
        img_domains.append(datum['domain'])
        print(ret_cap)
        
    
    #print('stop here')
    #pdb.set_trace()     
    return img_paths,img_caps,img_domains

####new###
  


for i in range(0,0):
    print("Item number: %6d"%i)
    start_time = time.time()
    text_query ='santa'    
    ret=search_and_save_one_query(text_query, i) 
    end_time = time.time()   
    pdb.set_trace()
    print("--- Time elapsed for 1 query: %s seconds ---" % (end_time - start_time))      
### Loop ####
'''for i in range(start_counter,end_counter,2):
    print("Item number: %6d"%i)
    start_time = time.time()
    ann = clip_data_annotations[i]
    text_query = visual_news_data_mapping[ann["id"]]["caption"]
    id_in_clip = i 
    image_id_in_visualNews = ann["image_id"]
    text_id_in_visualNews = ann["id"]
    new_folder_path = os.path.join(full_save_path,str(i))
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    search_and_save_one_query(text_query, id_in_clip, image_id_in_visualNews, text_id_in_visualNews, new_folder_path) 
    end_time = time.time()   
    print("--- Time elapsed for 1 query: %s seconds ---" % (end_time - start_time))'''

