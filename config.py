import os
import requests
from io import BytesIO
import shutil
import json  
import pdb
import numpy as np

from utils.common_utils import read_json_data,load_jsonl_file  
cleanprompt="You are a journalist. You write factual news article in English like on a News paper. Understand the text in terms of the specific details like named entities,  dates, location, event and  provide a concise factual summary mentioning these specific details.    Always try to understand and include the root cause of the event/protest/action/meeting from the text and hashtags used in the summary. Do not drop hashtags in the summary text, instead if needed, capture and include the general meaning of the hashtags in the summary text. The summary text must can be one only line. Only respond with the summary string and nothing else. Here is the input:" 
###############################################################common stuff
gkb_stop_words=['music','film','album','movie','club','actor']



        
dataset_name ='runtime'

if dataset_name=='runtime':
    region                   ='Global' 
    noisy_text               = False
    save_folder_path         ='runtime_data/'
    #save_folder_path         ='/data2/users/arka/rav/runtime_data/'
    split_type               ='live_run' 
    sub_split                ='eng'
    startdate                =''#'after:2023' #'before:2019'#
    llm                      = 'chatgpt'
    
    if region =='Global':
       dir_ser_api     ='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX' 
       gse_cid         ='XXXXXXXXXXXXXXXXXXXXXXX'
   
    #XV_DATA_DIR     ='/data2/users/arka/rav/'
    DATA_DIR        ='runtime_data/'
    ClAIM_DATA_DIR  ='runtime_data/'+split_type+'/claim_images/'
    
    #XV_DATA_DIR     ='/data2/users/arka/rav/'
    #DATA_DIR        ='/data2/users/arka/rav/runtime_data/'
    #ClAIM_DATA_DIR  ='/data2/users/arka/rav/runtime_data/'+split_type+'/claim_images/'
    placeholder_path='runtime_data/no-results.png'
    dictfile        ='runtime_data/runtime_chatgpt_res.npz'
    clean_dictfile  ='runtime_data/runtime_chatgpt_cleantext.npz'
    res_folder      ='runtime_data/res_chatgpt/'
    ann_fname       ='runtime_data/runtime_data_rimkser.npz'
    if noisy_text:
        cprompt   =cleanprompt
    else:
        cprompt   ='do_nothing' 
    
    txt_sim_thres            =.99
    gkb_thres                =0.44  
    







