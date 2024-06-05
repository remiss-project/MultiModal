import random
import os
import gradio as gr
#import tkinter as tk
#from tkinter import simpledialog
from utils.rav_utils      import download_and_display_image,load_jsonl_file, wgsearch,  getdatafrom_idxk
from utils.rev_ser        import reverse_search_and_save_one_query,reverse_getdatafromjson
from utils.img_sim_stuff  import compare_query_with_evidence
import math
import json    
from PIL import Image,ImageDraw
import pdb
import ast
import copy
import tempfile
from utils.chtgpt import get_response_ch_api_wrap
from utils.graph_stuff import get_graph_igc, get_graph_igc_conditioned,get_ent,zsal_vt,cleancheckupdate
from itertools import combinations
from utils.common_utils import read_json_data
from googletrans import Translator
import numpy as np
import requests
from io import BytesIO
import shutil
translator = Translator()
from transformers import ViTForImageClassification
vitmodel = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

from utils.config import ClAIM_DATA_DIR,                           placeholder_path, dataset_name,            ann_fname,test_samples#, node_similarity_threshold,  edge_similarity_threshold ,vis_sim_thres,txt_sim_thres  
max_run             = 3
max_ser_attempt     = 3
mode                = 'mauto'
overwrite_evd       = True
LLM_assist          = True
global ann_dict
ann_dict            = np.load(ann_fname,allow_pickle=True)
ann_dict            =ann_dict[ann_dict.files[0]]
ann_dict            =ann_dict.flat[0]

placeholder_img = Image.open(placeholder_path)
global temp_data
temp_data           = []
global vis_sim_thres       
global txt_sim_thres            
global node_similarity_threshold
global edge_similarity_threshold

########################


def remiss_sample():
    # Assuming test_samples is defined elsewhere in your code
    current_index = np.random.randint(len(test_samples), size=1)[0]
    cd = test_samples[current_index]
    text_input = cd['text']
    media = cd.get('media', [])
    v_claim_p = ClAIM_DATA_DIR + 'claim_' + str(cd['id']) + '_img.jpg'
 
    ##init
    img_input = placeholder_img 
    
    if media[0].get('type') =='photo':
       try:
             img_input, v_claim_p = download_and_display_image(media[0].get('url'), v_claim_p)
       except:
        #print (media)
        #try:
        #   img_input, v_claim_p = download_and_display_image(media[0].get('variants', [])[0].get('url'), v_claim_p)
        #except:
        print('giving up')
    return text_input, img_input
    
    
def image_grid(imgs):
    if len(imgs) <=3:
        cols = len(imgs)
        rows = 1
    else:
        cols = 3
        rows = math.ceil(len(imgs)/cols)
    imgs = [Image.open(img) for img in imgs]
    imgs += [Image.new('RGB', (500, 500))]*(cols*rows-len(imgs))
    imgs = [img.resize((500, 500)) for img in imgs]

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    draw = ImageDraw.Draw(grid)
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
        draw.rectangle((i%cols*w, i//cols*h, (i%cols+1)*w, (i//cols+1)*h), outline="black", width=2)

    return grid
def user_mark_annotation():
      #print('>>>>saving without evidence')
      user_given_best_t_sug=' user given best none '
      #root = tk.Tk()
      #root.withdraw()
      #user_given_best_t_sug = simpledialog.askstring("USER MARKS", "Enter Text Search Suggestion:")
      global ann_dict
      global temp_data
      [k, xv_idx, best_k, best_xv_idx, best_xvs, best_t_sug, found_flag, current_index, DUML, v_claim_p, DUMC, t_claim, v_claim, t_sum , qterms_sem, qterms_plc, qterms_fce, qterms_date, t_sug ,old_t_sug,xt,current_index]= temp_data
      temp_data=[k, xv_idx, best_k, best_xv_idx, best_xvs, best_t_sug, found_flag, current_index, DUML, v_claim_p, DUMC, t_claim, v_claim, t_sum , qterms_sem, qterms_plc, qterms_fce, qterms_date, user_given_best_t_sug ,old_t_sug,xt,current_index]

      print('<<---------------------------- USER changed seacrh term>')
def get_thres_user_input(node_thres_input_level,edge_thres_input_level,vis_sim_thres_level):

    node_sim_thres_list=[0.15, 0.35, 0.55, 0.75, 0.85]
    edge_sim_thres_list=[0.05, 0.15, 0.25, 0.55, 0.75]
    vis_sim_thres_list =[0.70, 0.80, 0.91, 0.95, 0.98]
    txt_sim_thres_list =[0.50, 0.60, 0.70, 0.85, 0.95]
    return node_sim_thres_list[node_thres_input_level],edge_sim_thres_list[edge_thres_input_level],vis_sim_thres_list[vis_sim_thres_level]
def  get_ergraph_stuff_from_text(text):
       graph_img, DG, res     = get_graph_igc( text)
       entities                           = get_ent(DG)
       qterms_sem                         = entities['EVENT']
       qterms_plc                         = entities['LOCATION']
       qterms_fce                         = entities['PERSON']
       qterms_date                        = entities['DATE']
       return  qterms_sem, qterms_plc, qterms_fce, qterms_date


def initcall(t_claim):
       ###          call llm for er graph  
       ####       call get ents to get entites like people place and sthit
       #####     call llm with text and er graph to get a steatrch string 
       ######
       
       ##

       qterms_sem, qterms_plc, qterms_fce, qterms_date  =  get_ergraph_stuff_from_text(t_claim)
       prompt=f" As a journalist assistant, your task is to understand news articles  from provided tweets, and generate effective search term sets to validate the information. The query text is '{t_claim}'. The search string is composed of elements 'action'  terms from {qterms_sem}, 'place' terms from {qterms_plc}, 'people' terms from {qterms_fce}, and 'date' terms from {qterms_date}.  Create a new search string with words from the lists of 'place', 'people', 'action', 'date' search terms and also the query text. Try to retain specific information like names of people, place, organization, date and other things you deem relevant from the query text and the search term lists. Make sure the number of words in the new 'search_string' is always between eight to ten words, DON'T VIOLATE THIS. Output this new 'search_string'.  Only respond as python dict of 1 element, 'search_string'. The All response should be in the original language of the input text and free of any special symbols like hashtags or emoticons. Output should be like this: '{{'search_string': search terms here}}'. Make sure the output is interpretable as ast.literal_eval. Now respond with this output dict and NOTHING ELSE." 
      
       datach               = get_response_ch_api_wrap(prompt+t_claim)
       datach               = ast.literal_eval(datach)
       t_sug                  = datach ['search_string']  
       v_cap                = 'no blip sorry'
       
      
       return qterms_sem, qterms_plc, qterms_fce, qterms_date,t_sug , v_cap 
    
def retrysearchterm():
    global temp_data
    k, xv_idx, best_k, best_xv_idx, best_xvs, best_t_sug, found_flag, current_index, label, v_claim_p ,v_cap, t_claim, v_claim,  t_sum , qterms_sem, qterms_plc, qterms_fce, qterms_date, t_sug ,old_t_sug,xt,ci=temp_data

    suc=1
    t_sug_new=' '
    while(1):
         try:
                  prompt=f"As a journalist assistant, your task is to understand news articles  from provided tweets, and generate effective search term sets to validate the information. The query text is '{t_sum}'. The search string is composed of elements 'action'  terms from {qterms_sem}, 'place' terms from {qterms_plc}, 'people' terms from {qterms_fce}, and 'date' terms from {qterms_date}. This was the 'old_search_string': '{t_sug}' used to search the evidence image which returned no hits, maybe because it is too verbose or includes obscure search terms. Create a new search string with words from the lists of 'place', 'people', 'action', 'date' search terms and also the query text. Try to retain specific information like names of people, place, organization, date and other things you deem relevant from the query text and the search term lists. Make sure the number of words in the new 'search_string' is always between eight to ten words, DON'T VIOLATE THIS. Output this new 'search_string'. The new 'search_string' must be different from 'old_search_string', but similar to the query text. Only respond as python dict of 1 element, 'search_string'. The All response should be in the original language of the input text and free of any special symbols like hashtags or emoticons. Output should be like this: '{{'search_string': search terms here}}'. Make sure the output is interpretable as ast.literal_eval.Now respond with this output dict and NOTHING ELSE." 
         
                  new_search_string      = get_response_ch_api_wrap(prompt)
         except:
                  pdb.set_trace()
         try:
                  t_sug_new              = ast.literal_eval(new_search_string)['search_string']
                  suc=0
         except:
                  print('rephrasing ST  LLM failed, retrying ...')
         if suc==0:
                   break   
    temp_data=[ k, xv_idx, best_k, best_xv_idx, best_xvs, best_t_sug, found_flag, current_index, label, v_claim_p , v_cap,t_claim, v_claim,  t_sum , qterms_sem, qterms_plc, qterms_fce, qterms_date, t_sug_new ,old_t_sug,xt,ci]
    
    return t_sug_new  
    

def update_resp_rec(  scr_values ,xv_text,t_claim,t_sug):
             
    [sem, plc, fce, obj, gcp, scetxt, fcp1]  =scr_values
    claim_graph_img, CDG, cres     = get_graph_igc( t_claim)
    tcres                          = copy.deepcopy(cres)
 
    
    xevd_graph_img,  XDG, xres     = get_graph_igc_conditioned(xv_text,t_claim,tcres)
    t_sum=t_claim


    #pdb.set_trace()
    try:
      prompt=f"Given an image 'claim_v',   text   'claim_t'.    Our task is find an image 'evidence_xv' that is similar to  'claim_v'  by searching the internet with search terms from the 'claim_t'.  The idea is to  iteratively improve upon the search terms based on the retrieved evidence.   The  text   'claim_t'  is '{t_sum}'.   This was the old search term 'search_old'  : '{t_sug}' , which lead to an 'evidence_xv'  that  has  caption 'xv_cap' : '{xv_text}' .  The similarity between  'claim_v' and 'evidence_xv' is  {sem}  in terms of visual semantics, {plc}  in terms of  place or location visuals , {fce} in terms of faces, {obj} in terms of visual objects , {scetxt} in terms of scene text  and {gcp} in terms of caption generated of the image.    The graph for  'claim_t' is 'cres' is {cres}. The  graph for the evidence 'xv_cap' is 'xres' is {xres}.  Find out the nodes and edges in 'cres' which are not present in 'xres'.  Using the visual scores and th graphs you can tell which parts of   'claim_t'  are not being addresssed and focus the next  seach term on those lines.   In summary find the nodes and relationships of 'cres' about which 'xres' tells us nothing, and use these unmatched edges to form the next search string. The search string can have words only from  the 'old_search_string' and 'claim_t'. It is important to retain date and other specifics from 'claim_t'  and 'cres'  in the new 'search_string'. The  'search_string' is used to search the internet, so if there are too many words the results are often confusing, so please make sure the search string has eight to ten words. Output this new 'search_string'. Output can not be same as input, 'search_string' must be slighly different from 'old_search_string' while remaining similar to  'claim_t'  . Only respond as python dict of 1 element, 'search_string'. The All response should be in the orignal language of the input text and free of any special symbols like hashtags or emoticons. Make sure the output is interpretable as ast.literal_eval. Now respond with only this output dict and NOTHING ELSE. "  
    except:
          pdb.set_trace()
    
    ###>>>>
    #pdb.set_trace()
    generated_search_term         = get_response_ch_api_wrap(prompt)
    generated_search_term         = ast.literal_eval(generated_search_term)
    generated_search_term         = generated_search_term ['search_string']
      

    print('####################################################################################################################')
    print(t_sug)
    print(prompt)
    print(generated_search_term)
    print('####################################################################################################################')
    #############
       
    return generated_search_term     


########################



def generate_18_digit_number():
    return random.randint(10**17, 10**18 - 1)


def create_empty_image(width, height, background_color=(255, 255, 255)):
    # Create a new image with the specified width, height, and background color
    image = Image.new("RGB", (width, height), background_color)
    return image

def load_claim(img_claim, text_claim,node_thres_input_level,edge_thres_input_level,vis_sim_thres_input_level):
        v_claim               = img_claim
        t_claim               = text_claim
        current_index         = generate_18_digit_number()
        file_name             = str(current_index)+'.jpg'
        v_claim_p             = os.path.join(ClAIM_DATA_DIR, file_name) 
        img                   = v_claim
        img.save(v_claim_p)  

        print('<no evidence>   ; not annotated yet')
        k, xv_idx, best_k, best_xv_idx, best_xvs =[0,0,0,0,0]
        b_vis_evidence        = placeholder_img
        b_vis_evidence_cap, best_t_sug, t_sug, old_t_sug, t_sum, qterms_sem, qterms_plc, qterms_fce, qterms_date     = ['','','','','','','','','']
        
        if  LLM_assist==True:
                      qterms_sem, qterms_plc, qterms_fce, qterms_date,t_sug,v_cap  = initcall(t_claim)
                      t_sum=t_claim  

        new_CDG               = create_empty_image(200,200)       
        found_flag            = 'not found'
        
        
        #######
        ##rev search stuff
        #pdb.set_trace()
        rev_json=reverse_search_and_save_one_query(v_claim_p, current_index)
        ent,ent_scr,bgl,xt=reverse_getdatafromjson(rev_json)
        #xt=['no_text_evidence','no_text_evidence']
        #####
        
        
        global temp_data
        temp_data =   [k, xv_idx, best_k, best_xv_idx, best_xvs, best_t_sug, found_flag, current_index, '404', v_claim_p, v_cap, t_claim, v_claim, t_sum , qterms_sem, qterms_plc, qterms_fce, qterms_date, t_sug ,old_t_sug,xt,current_index] 
        
        
        ###<<setttig thresholds for this run
        nth,eth,vsth= get_thres_user_input(int(node_thres_input_level),int(edge_thres_input_level),int(vis_sim_thres_input_level))
        global node_similarity_threshold
        node_similarity_threshold=nth
        global edge_similarity_threshold
        edge_similarity_threshold=eth    
        global vis_sim_thres
        vis_sim_thres=vsth    

        #pdb.set_trace()
        
        return new_CDG,label,v_claim,t_claim,b_vis_evidence,0 ,b_vis_evidence_cap,  b_vis_evidence, 0, b_vis_evidence_cap,  t_sug ,old_t_sug,found_flag, xt,current_index    
   


##

def find_imgsim_scores_topk(ransamidx,v_claim_p, t_claim, xv_p, xv_cap, k, xt):

    #vs,vs_l,vts,ts              = compare_query_with_evidence(img_path, caps, vis_evidence_caps, vis_evidence_paths,text_evidences) 
    #xv_s, xv_ls, xv_cats, xvcap_s = compare_query_with_vis_evidence(v_claim_p, t_claim, xv_cap, xv_p)
    
    xv_s, xv_ls, xv_cats, xvcap_s,ts = compare_query_with_evidence(v_claim_p, t_claim, xv_cap, xv_p,xt)
    #pdb.set_trace()
    xt_text =xt[np.argmax(ts)]
      
    
    xv_scores = [max(xv, xvcap) for xv, xvcap in zip(xv_s, xvcap_s)]
    xv_idxs = np.argsort(xv_scores)[-k:][::-1]

    new_CDGs, new_VDGs,new_TDGs ,found_flags, xv_s_list, scr_values_list, xv_texts, xv_cats_list,vtx_val_list,ttx_val_list = [], [], [], [], [], [], [],[],[],[]
    try:
     for xv_idx in xv_idxs:
        xv         =Image.open(xv_p[xv_idx])
        xv_text    =         xv_cap[xv_idx]
        scr_values =          xv_ls[xv_idx]
        xv_cat     =        xv_cats[xv_idx]

        found_flag = 'found' if any(all(value > vis_sim_thres for value in combo) for combo in combinations(scr_values[0:6], 3)) else 'not found'
        ser_terms, new_CDG, new_VDG, vtx_val,new_TDG, ttx_val  = zsal_vt(ransamidx,    v_claim_p,       t_claim, image_grid(xv_p),xv_p[xv_idx], xv_text,xt_text, vis_sim_thres, node_similarity_threshold,  edge_similarity_threshold, xv_s[xv_idx],scr_values,0)
        
        new_CDGs.append(new_CDG)
        new_VDGs.append(new_VDG)
        vtx_val_list.append(vtx_val)
        new_TDGs.append(new_TDG)
        ttx_val_list.append(ttx_val)
        found_flags.append(found_flag)
       
        xv_s_list.append(xv_s[xv_idx])
        scr_values_list.append(scr_values)
        xv_texts.append(xv_text)
        xv_cats_list.append(xv_cat)
        
    except:
     pdb.set_trace() 

    return new_CDGs, new_VDGs,new_TDGs ,found_flags, xv_idxs.tolist(), xv_s_list, scr_values_list, xv_texts, xv_cats_list,vtx_val_list,ttx_val_list

       
          
def get_evidence():
      global temp_data
      #---------------------------------------------------------------------------#
      [k, xv_idx, best_k, best_xv_idx, best_xvs, best_t_sug, found_flag, current_index, DUML, v_claim_p, v_cap, t_claim, v_claim, t_sum , qterms_sem, qterms_plc, qterms_fce, qterms_date, t_sug ,old_t_sug,xt,current_index]=temp_data 
      
      #---------------------------------------------------------------------------#
      print('<--------loading done---------------------->')  
      ser_attempt=0
      found_sev  =1 
      while(ser_attempt < max_ser_attempt):
              print('sysmsg[-][-] <   search try    '+str(ser_attempt)+' '+ str(t_sug)+'                >')
              suc, xv_p_list,xv_cap_list,xv_dom_list =  wgsearch(t_sug,current_index,k)
              if len(xv_p_list)<1:
                  print('sysmsg[-][-][-] ----<  bad search term  :'+str(t_sug)+ ' >>> retrying with ')
                  if LLM_assist==True:
                           t_sug= retrysearchterm() 
                  else:
                           print('this search term failed ')
                           t_sug=t_sug+ 'retrying '
                  
                  print(t_sug)
                  print('sysmsg[-][-][-] >---')
              else:
                  found_sev=0
                  break
              ser_attempt+=1  

      print('------->>>>>>') 
      
      ############################################################################################################
      topk=3
      if  found_sev==1: 
          print('ALERT: GOOGLE SEARCH Failed-------------------')
          new_CDG                                                  = create_empty_image(200,200) 
          new_VDG                                                  = create_empty_image(200,200) 
          xv_p_topklist             =[placeholder_path] * topk
          xv_cap_topklist           =['']           * topk
          xv_dom_topklist           =['NE']         * topk
          new_CDG_topklist          =[new_CDG ]     * topk
          new_VDG_topklist          =[new_VDG ]     * topk
          found_flag_c_topklist     =['not found']  * topk
          xv_idx_c_topklist         =[0]            * topk
          xvs_c_topklist            =[0]            * topk
          scr_values_c_topklist     =[0]            * topk
          xv_text_topklist          =['no xvtext']  * topk 
          xv_cat_topklist           =['no cat']     * topk
          xv_gsval_topklist         =[0]            * topk
      else:                         
          print('something found !!')
          new_CDG_topklist,new_VDG_topklist,new_TDG_topklist,found_flag_c_topklist,  xv_idx_c_topklist,  xvs_c_topklist,   scr_values_c_topklist , xv_text_topklist,xv_cat_topklist,xv_gsval_topklist,xt_gsval_topklist   = find_imgsim_scores_topk(current_index,v_claim_p ,t_claim,xv_p_list,xv_cap_list,topk,xt)
          
          #print('here now 1')
          #pdb.set_trace()
          
          xv_dom_topklist= [xv_dom_list[x]        for x in  xv_idx_c_topklist ]
          xv_p_topklist  = [xv_p_list[x]          for x in  xv_idx_c_topklist ]
          xv_topklist    = [Image.open(xv_p)      for xv_p in xv_p_topklist]

      
      

      #------------------------------------------annotation update------------------------------------------# 
      new_CDG,new_VDG,found_flag_c,  xv_idx_c,  xvs_c,   scr_values_c , xv_text,xv_cat,xv_gsval=new_CDG_topklist[0],new_VDG_topklist[0],found_flag_c_topklist[0],  xv_idx_c_topklist[0],  xvs_c_topklist[0],   scr_values_c_topklist[0] , xv_text_topklist[0],xv_cat_topklist[0],xv_gsval_topklist[0]
      old_t_sug=t_sug
      if ((xvs_c >= best_xvs) or (found_flag_c=='found')) :#and (found_flag!='found'):                      
              print('better found ..')
              found_flag                          =  found_flag_c 
              best_xvs                            =  xvs_c
              best_t_sug                          =  t_sug
              best_xv_idx                         =  xv_idx_c 
              best_k                              =  k
              ann_dict[current_index]             =  {'ck':best_k,'bk':best_k,'idx':best_xv_idx,'status':found_flag,'best_t_sug':best_t_sug,'sim_s':xvs_c,'best_sim_s':best_xvs}
      else:
           ann_dict[current_index]['ck']    = k
           ann_dict[current_index]['sim_s'] =xvs_c

      if  found_flag =='found' or found_flag =='found VOOC' :
                    print('sysmsg[-] <  '+str(found_flag)+'>        <done> ')
      else:
            if k < max_run and (found_sev==0):
                  if LLM_assist==True:           
                       t_sug = update_resp_rec(scr_values_c,xv_text,t_claim,old_t_sug) 
                  else:
                       t_sug=t_sug + '<resp_rec>'
            else:
                 print('tried '+str(k)+' times cumulative, or found, not more runs for this one ')
                 print('sysmsg[-] <  '+str(found_flag)+'>         <done>')
                 #running_status='new'
                 #save_annotation()
      
      k=k+1
      temp_data =   [k, xv_idx, best_k, best_xv_idx, best_xvs, best_t_sug, found_flag, current_index, '404', v_claim_p, v_cap, t_claim, v_claim, t_sum , qterms_sem, qterms_plc, qterms_fce, qterms_date, t_sug ,old_t_sug,xt,current_index]    
      #------------------------------------------annotation update------------------------------------------#    
      
      
      
      return new_CDG_topklist,new_VDG_topklist,new_TDG_topklist,found_flag_c_topklist,  xv_idx_c_topklist,  xvs_c_topklist,   scr_values_c_topklist , xv_text_topklist,xv_cat_topklist,xv_gsval_topklist, xv_dom_topklist,xv_p_topklist,xv_topklist
          


def get_best_evidence():
       global temp_data
       new_CDG_topklist,new_VDG_topklist,new_TDG_topklist,found_flag_c_topklist,  xv_idx_c_topklist,  xvs_c_topklist,   scr_values_c_topklist , xv_text_topklist,xv_cat_topklist,xv_gsval_topklist, xv_dom_topklist,xv_p_topklist,xv_topklist =get_evidence()
       new_CDG0,new_VDG0,new_TDG0,new_dom0,xv_cat0,xv_text0,xvs0,found_flag0,xv_img0=new_CDG_topklist[0],new_VDG_topklist[0], new_TDG_topklist[0], xv_dom_topklist[0],xv_cat_topklist[0],xv_text_topklist[0],xvs_c_topklist[0], found_flag_c_topklist[0],xv_topklist[0]
       new_CDG1,new_VDG1,new_TDG1,new_dom1,xv_cat1,xv_text1,xvs1,found_flag1,xv_img1=new_CDG_topklist[1],new_VDG_topklist[1], new_TDG_topklist[1], xv_dom_topklist[1],xv_cat_topklist[1],xv_text_topklist[1],xvs_c_topklist[1], found_flag_c_topklist[1],xv_topklist[1]
       
       
       [k, xv_idx, best_k, best_xv_idx, best_xvs, best_t_sug, found_flag, current_index, DUML, v_claim_p, DUMC, t_claim, v_claim, t_sum , qterms_sem, qterms_plc, qterms_fce, qterms_date, t_sug ,old_t_sug,xt,current_index]= temp_data
       return new_CDG0,new_VDG0,new_TDG0,new_dom0,xv_cat0,xv_text0,xvs0,xv_img0,new_CDG1,new_VDG1,new_TDG1,new_dom1,xv_cat1,xv_text1,xvs1, xv_img1, t_claim, v_claim, found_flag, current_index, t_sug ,old_t_sug
       
   
    
    
    
    
    

with gr.Blocks(theme=gr.themes.Default(spacing_size="sm", radius_size="none", text_size="lg")) as app:
    with gr.Row():
        gr.Markdown("Image Text Pair Search")

    with gr.Row():
        with gr.Column():
            sload = gr.Button(value="start")
            gr.Markdown("### CLAIM")
            img_input = gr.Image(type='pil', label="Input Image")
            text_input = gr.Textbox(label="Input Text")
            
            load_sample_button = gr.Button(value="Load Sample")
            
            node_thres_input_level=gr.Slider(minimum=0, maximum=5, step=1, label="Entity Similarity")
            edge_thres_input_level=gr.Slider(minimum=0, maximum=5, step=1, label="Action Similarity")
            vis_sim_thres_input_level=gr.Slider(minimum=0, maximum=5, step=1, label="Visual Similarity")
            
            #gr.Markdown("### claim parts verified")
            img_claim = gr.Image(type='pil', label="Claim Image",visible=False)
            text_claim = gr.Textbox(label="Claim Text",visible=False)
            new_CDG = gr.Image(type='pil', label="New Claim Text Graph",visible=False)
            label = gr.Textbox(label="Label",visible=False)
            
            usermark = gr.Button(value='Change Search Term')       
            gr.Markdown("### STATUS")
            
            
            t_sug = gr.Textbox(label="Next Suggested Search Term")
            old_t_sug = gr.Textbox(label="Generated by Search Term")
            found_flag = gr.Textbox(label="Found Status")
            ci = gr.Textbox(label="Index")
            
        with gr.Column():
            run = gr.Button(value="Run")
            gr.Markdown("### TEXT EVIDENCE")   
            xt    = gr.Textbox(label="XT from Reverse Search with Image")
            with gr.Row(): 
              with gr.Column(): 
                new_TDG0 = gr.Image(type='pil', label="Text Evidence Graph")
                new_CDG0 = gr.Image(type='pil', label="Claim text Graph")
                
                gr.Markdown("### VIS EVIDENCE 1 ")   
                
                new_VDG0 = gr.Image(type='pil', label="Visual Evidence Graph")
                
                xv_img0           = gr.Image(type='pil', label="Current Visual Evidence")
                xvs0         = gr.Textbox(label="Visual Similarity Cumulative")
                xv_cat0            = gr.Textbox(label="Top Channels of Image Match")
                xv_text0 = gr.Textbox(label="Visual Evidence Caption")
                new_dom0    =gr.Textbox(label="Visual Evidence domain")  
                
              
              with gr.Column(): 
                new_TDG1 = gr.Image(type='pil', label="Text Evidence Graph")
                new_CDG1 = gr.Image(type='pil', label="Claim text Graph")   
                
                gr.Markdown("### VIS EVIDENCE 2 ")   
                
                new_VDG1 = gr.Image(type='pil', label="Visual Evidence Graph")
                
                xv_img1           = gr.Image(type='pil', label="Current Visual Evidence")
                xvs1         = gr.Textbox(label="Visual Similarity Cumulative")
                xv_cat1            = gr.Textbox(label="Top Channels of Image Match")
                xv_text1 = gr.Textbox(label="Visual Evidence Caption")
                new_dom1    =gr.Textbox(label="Visual Evidence domain")  
                
                
               



            

    sload.click(load_claim, inputs=[img_input, text_input,node_thres_input_level,edge_thres_input_level,vis_sim_thres_input_level], 
    outputs=[new_CDG, label, img_claim, text_claim, xv_img0, xvs0, xv_text0, xv_img0, xvs0, xv_text0, t_sug, old_t_sug, found_flag,xt,ci])   
       
    load_sample_button.click(remiss_sample, outputs=[text_input, img_input])


    run.click(get_best_evidence, inputs=None, outputs=[new_CDG0,new_VDG0,new_TDG0,new_dom0,xv_cat0,xv_text0,xvs0,xv_img0,new_CDG1,new_VDG1,new_TDG1,new_dom1,xv_cat1,xv_text1,xvs1, xv_img1, text_claim,img_claim, found_flag, ci, t_sug ,old_t_sug])
    

    
    usermark.click(user_mark_annotation,inputs=None, outputs=None)        

app.launch(ssl_verify=False, ssl_keyfile="key.pem", ssl_certfile="cert.pem")




