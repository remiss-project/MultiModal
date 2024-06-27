import base64
import requests
from io import BytesIO
import gradio as gr
from gradio_client import Client
from PIL import Image
from collections import Counter
import tempfile
import os
import pdb
import numpy as np
import subprocess
#import paramiko 
import json



bcn19 = [ 239514, 23956, 241514, 2430, 243049, 243340, 244044,
    244224, 25, 2501, 254917, 259742, 261551, 262251, 263653, 266243, 268688, 2697,
    2703, 2716, 272961, 273018, 274457, 274612, 276477, 277660, 277769, 2785, 28295,
    28316, 284931, 285048, 2880, 288030, 2894, 290897, 2969, 297137, 298035, 298369,
    2987, 299617, 299719, 307, 3201, 3207, 337, 34614, 3588, 3777, 4123, 4412, 47,
    47310, 4767, 47795, 48, 49, 49219, 5029, 506, 511, 519, 52, 544, 54792, 57072,
    5785, 58423, 60394, 60630, 63, 63453, 63729, 65420, 66, 66403, 670, 68338, 6858,
    710, 7207, 7348, 74419, 74605, 75189, 75224, 755, 7620, 77110, 787, 80, 80292,
    8289, 83498, 8395, 84374, 8481, 84825, 85687, 85710, 86051, 8691, 88085, 88212,
    9031, 90376, 90573, 9085, 91277, 92, 93, 9423, 9439, 94880, 95535, 9632, 96581,
    97835, 9846, 99, 99157,102651, 103789, 105187, 109124, 11100, 11109, 111173, 112615, 112643, 11275,
    113, 113012, 11434, 115509, 116627, 117797, 119912, 120359, 123066, 124445,
    12454, 128, 128587, 130112, 131, 1316, 132, 133150, 133711, 135, 137452,
    138329, 141274, 141997, 142, 143059, 147722, 149735, 150264, 15397, 154392,
    154918, 155797, 15619, 156713, 158080, 16, 1624, 163610, 163866, 164099,
    164941, 166139, 167270, 17318, 174078, 175577, 17962, 186999, 188616, 1902,
    19110, 19276, 193417, 194021, 19759, 197878, 198381, 2006, 200990, 207023,
    207765, 212563, 212696, 214335, 215028, 217109, 21859, 220949, 220963, 22136,
    222895, 222923, 225443, 22906, 231587, 231591, 23184, 23255, 232574, 232770,
    233004, 234418, 235191]
    


    
jsonl_files     = ['remiss_data/remiss_data_complete/barcelona_2019_search-1.media.jsonl', 'remiss_data/remiss_data_complete/generales_2019_search-1.media.jsonl', 'remiss_data/remiss_data_complete/generalitat_2021_search-1.media.jsonl', 'remiss_data/remiss_data_complete/MENA_Agressions.media.jsonl', 'remiss_data/remiss_data_complete/MENA_Ajudes.media.jsonl', 'remiss_data/remiss_data_complete/Openarms.media.jsonl']

splits          = ['bcn19','gen19','gen21','mena_aggr','mena_ajud','openarms']
startdates      = [' before:2020-01-01','before:2020-01-01','before:2022-01-01','before:2020-01-01','before:2020-01-01','before:2020-01-01']
split_to_index = {split: idx for idx, split in enumerate(splits)}


global split_type
split_type='undefined'
global sample_idx
sample_idx=404
global gt_label
gt_label=2

current_local_dir_path='/data/users/arka/rav/remiss_data/'

client = Client("https://158.109.8.113:7860/", ssl_verify=False,auth=("remiss", "t6y7u8?"))
placeholder_path= current_local_dir_path + 'no-results.png'
placeholder_img = Image.open(placeholder_path)




PRISTINE_LABEL=0
FAKE_LABEL=1
labelnames=['PRISTINE','FAKE']
global user_labelled
#user_labelled=np.load('/data/users/arka/rav/remiss_data/outputs/user_input_labels.npz',allow_pickle=True)
user_labelled=np.load(current_local_dir_path +'outputs/user_input_labels.npz',allow_pickle=True)
user_labelled=user_labelled[user_labelled.files[0]]
user_labelled=user_labelled.flat[0]


###################################################################################################################
def user_input_labels_PRISTINE():
    global split_type
    global sample_idx
    global user_labelled
    key= str(split_type) + '< >' + str(sample_idx)
    user_labelled[key]=PRISTINE_LABEL
    np.savez(current_local_dir_path+'outputs/user_input_labels.npz',user_labelled=user_labelled)

def user_input_labels_FAKE():
    global split_type
    global sample_idx
    global user_labelled
    key= str(split_type) + '< >' + str(sample_idx)
    user_labelled[key]=FAKE_LABEL

    np.savez(current_local_dir_path+'outputs/user_input_labels.npz',user_labelled=user_labelled)



def load_jsonl_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = [json.loads(line) for line in lines]
    return data

        
def remiss_sample_already_stats(jidx):
    # Assuming test_samples is defined elsewhere in your code
    #
    jsonl_file      = jsonl_files[jidx]
    
    global split_type
    split_type      = splits[jidx]
    
    startdate       = startdates[jidx]
    test_samples    = load_jsonl_file(jsonl_file)
    data_path       = current_local_dir_path+'remiss_data_complete/' + split_type 
    vl_dict         = np.load(data_path + '/remiss_filter_marked.npz', allow_pickle=True)
    vl_dict         = vl_dict[vl_dict.files[0]]
    vl_dict         = vl_dict.flat[0] 
    
    output_base_folder = 'remiss_data/outputs/'+  split_type 
    # List all folders in the base directory
    all_folders = [os.path.join(output_base_folder, d) for d in os.listdir(output_base_folder) if os.path.isdir(os.path.join(output_base_folder, d))]
    # Filter only non-empty folders
    non_empty_folders = [folder for folder in all_folders if os.listdir(folder)]
    # Process each non-empty folder
    channel_counter = Counter()
    pdb.set_trace()
    for output_folder in non_empty_folders:
         try:
            res=load_from_folder(output_folder)
            xv_cat=eval(res[4])[0]
            print(xv_cat)
            channel_counter.update([xv_cat])
         except:
            print('x')
    return channel_counter       

def load_remiss_sample_already(jidx):
    # Assuming test_samples is defined elsewhere in your code
    #
    jsonl_file      = jsonl_files[jidx]
    
    global split_type
    split_type      = splits[jidx]
    
    global user_labelled
    
    
    startdate       = startdates[jidx]
    test_samples    = load_jsonl_file(jsonl_file)
    data_path       = current_local_dir_path+'/remiss_data_complete/' + split_type 
    vl_dict         = np.load(data_path + '/remiss_filter_marked.npz', allow_pickle=True)
    vl_dict         = vl_dict[vl_dict.files[0]]
    vl_dict         = vl_dict.flat[0] 


    
    
    while True:
        current_index = np.random.randint(len(test_samples), size=1)[0]
        key= str(split_type) + '< >' + str(current_index)
        output_folder = f'remiss_data/outputs/{split_type}/{current_index}'
        if  (os.path.exists(output_folder) and os.listdir(output_folder)) and key in user_labelled.keys() :
            break
    global gt_label
    #pdb.set_trace()
    gt_label=labelnames[user_labelled[key]]
    global sample_idx
    sample_idx = current_index
    #output_folder = f'outputs/{split_type}/{sample_idx}/'
    #pdb.set_trace()
    return load_from_folder() + (gt_label,)
    

def load_from_folder(givenfolder=None):
    global sample_idx
    global split_type
    if givenfolder:
       output_folder=givenfolder  
    else:
       output_folder = f'remiss_data/outputs/{split_type}/{sample_idx}/'
    
    json_path = os.path.join(output_folder, 'data.json')
    
    if not os.path.exists(json_path):
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)

    def load_image(name):
        path = os.path.join(output_folder, name)
        return Image.open(path)

    new_CDG0 = load_image('graph_claim.png')
    new_VDG0 = load_image('graph_evidence_vis.png')
    new_TDG0 = load_image('graph_evidence_text.png')
    xv_img0 = load_image('evidence_image.png')
    new_CDG1 = load_image('graph_claim1.png')
    new_VDG1 = load_image('graph_evidence_vis1.png')
    new_TDG1 = load_image('graph_evidence_text1.png')
    xv_img1 = load_image('evidence_image1.png')
    v_claim = load_image('claim_image.png')
    


    

    # Convert relevant values to their appropriate types
    def convert_to_float(value):
        try:
            return float(value)
        except ValueError:
            return None
    t_claim   = data["claim_text"] 
    t_sug     = data["t_sug"]
    old_t_sug = data["old_t_sug"]
    current_index = data["id_in_json"]
    found_flag =data["found_flag"]
    # Example conversion for similarity scores
    xt     = data["text_evidence"]
    xt_scr = convert_to_float(data.get("text_evidence_similarity_score", 0.0))
    xt_gs0 = convert_to_float(data.get("text_evidence_graph_similarity_score", 0.0))
    
    new_dom0 = data["visual_evidence_domain"]
    xv_cat0  = data["visual_evidence_matched_categories"]
    xv_text0 = data["visual_evidence_text"]
    xvs0     = convert_to_float (data.get("visual_evidence_similarity_score", 0.0))
    xv_gs0 = convert_to_float(data.get("visual_evidence_graph_similarity_score", 0.0)) 
    
    new_dom1 = data["visual_evidence_domain1"]
    xv_cat1  = data["visual_evidence_matched_categories1"]
    xv_text1 = data["visual_evidence_text1"]
    xvs1     = convert_to_float (data.get("visual_evidence_similarity_score1", 0.0))
    xv_gs1 = convert_to_float(data.get("visual_evidence_graph_similarity_score1", 0.0)) 
    
    found_flag ='NOT found'
    ###MODIFY FOUND FLAG, INFERRING DECISION FROM GRAPH SCORES:
    if xt_gs0> 0.33  :
                  found_flag ='found'
    if xt=='no_text_evidence'  and ( (xv_gs0 > 0.33 and xvs0 > 0.8) or (xv_gs0 > 0.1 and xvs0 > 0.9)) :
                  found_flag ='found'
    
    return new_CDG0,new_VDG0,new_TDG0,new_dom0,xv_cat0,xv_text0,xvs0,xv_img0,xv_gs0,xt_gs0, new_CDG1,new_VDG1,new_TDG1,new_dom1,xv_cat1,xv_text1,xvs1, xv_img1,xv_gs1,xt_gs0, t_claim, v_claim, found_flag, current_index, t_sug ,old_t_sug,xt,xt_scr
    


def save_outputs_to_folder(new_CDG0, new_VDG0, new_TDG0, new_dom0, xv_cat0, xv_text0, xvs0, xv_img0, xv_gs0, xt,xt_scr, xt_gs0, new_CDG1, new_VDG1, new_TDG1, new_dom1, xv_cat1, xv_text1, xvs1, xv_img1, xv_gs1, xt_gs1, text_claim, img_claim, found_flag, ci, t_sug, old_t_sug):
    global sample_idx
    output_folder = 'remiss_data/outputs/'+split_type+'/'+str(sample_idx)
    os.makedirs(output_folder, exist_ok=True)

    # Save the images
    image_paths = {}
    def save_image(image, name):
        path = os.path.join(output_folder, name)
        image.save(path)
        image_paths[name] = path

    save_image(new_CDG0,  'graph_claim.png')
    save_image(new_VDG0,  'graph_evidence_vis.png')
    save_image(new_TDG0,  'graph_evidence_text.png')
    save_image(xv_img0,   'evidence_image.png')
    save_image(new_CDG1,  'graph_claim1.png')
    save_image(new_VDG1,  'graph_evidence_vis1.png')
    save_image(new_TDG1,  'graph_evidence_text1.png')
    save_image(xv_img1,   'evidence_image1.png')
    save_image(img_claim, 'claim_image.png')

    # Create a dictionary with the text and numeric values
    data = {
        "visual_evidence_domain"                 : new_dom0,
        "visual_evidence_matched_categories"     : xv_cat0,
        "visual_evidence_text"                   : xv_text0,
        "visual_evidence_similarity_score"       : xvs0,
        "visual_evidence_graph_similarity_score" : xv_gs0,
        "text_evidence"                          : xt,
        "text_evidence_similarity_score"         : xt_scr,
        "text_evidence_graph_similarity_score"   : xt_gs0,
        "visual_evidence_domain1"                : new_dom1,
        "visual_evidence_matched_categories1"    : xv_cat1,
        "visual_evidence_text1"                  : xv_text1,
        "visual_evidence_similarity_score1"      : xvs1,
        "visual_evidence_graph_similarity_score1": xv_gs1,
        "claim_text" : text_claim,
        "found_flag" : found_flag,
        "id_in_json" : sample_idx,
        "t_sug"      : t_sug,
        "old_t_sug"  : old_t_sug,
        "image_paths": image_paths
    }

    # Save the dictionary to a JSON file
    data = {k: v.tolist() if isinstance(v, np.int64) else v for k, v in data.items()}
    json_path = os.path.join(output_folder, 'data.json')
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Outputs saved to folder: {output_folder}")
    return output_folder

def save_outputs(new_CDG0, new_VDG0, new_TDG0, new_dom0, xv_cat0, xv_text0, xvs0, xv_img0, xv_gs0, xt, xt_scr, xt_gs0, new_CDG1, new_VDG1, new_TDG1, new_dom1, xv_cat1, xv_text1, xvs1, xv_img1, xv_gs1, xt_gs1, text_claim, img_claim, found_flag, ci, t_sug, old_t_sug):
    folder = save_outputs_to_folder(new_CDG0, new_VDG0, new_TDG0, new_dom0, xv_cat0, xv_text0, xvs0, xv_img0, xv_gs0, xt, xt_scr, xt_gs0, new_CDG1, new_VDG1, new_TDG1, new_dom1, xv_cat1, xv_text1, xvs1, xv_img1, xv_gs1, xt_gs1, text_claim, img_claim, found_flag, ci, t_sug, old_t_sug)
    return f"Outputs saved to folder: {folder}"


def remiss_sample(jidx):
    jsonl_file      = jsonl_files[jidx]
    
    global split_type
    split_type      = splits[jidx]
    
    startdate       = startdates[jidx]
    test_samples    = load_jsonl_file(jsonl_file)
    data_path       = 'remiss_data/remiss_data_complete/'+split_type 
    vl_dict         = np.load(data_path+'/remiss_filter_marked.npz',allow_pickle=True)
    vl_dict         = vl_dict[vl_dict.files[0]]
    vl_dict         = vl_dict.flat[0] 
    while(True):
        current_index = np.random.randint(len(test_samples), size=1)[0]
        if (current_index in vl_dict.keys()) and (not os.path.exists('outputs/'+split_type+'/'+str(current_index))) and vl_dict[current_index]['vt_score']>0.10 and vl_dict[current_index]['img_class']!=916:
           break 
    if split_type=='bcn19':
       while(True):
         current_index = np.random.randint(len(test_samples), size=1)[0]
         if (current_index in bcn19) and (current_index in vl_dict.keys()) and (not os.path.exists('outputs/'+split_type+'/'+str(current_index))) and vl_dict[current_index]['vt_score']>0.10 and vl_dict[current_index]['img_class']!=916:
             break         
           
    global sample_idx
    sample_idx=current_index       
    cd = test_samples[current_index]
    text_input = cd['text']
    media = cd.get('media', [])
    v_claim_p = data_path + '/claim_images/'+'claim_' + str(cd['id']) + '_img.jpg'
    img_input = Image.open(v_claim_p)
    return text_input, img_input

def create_empty_image(width, height, background_color=(255, 255, 255)):
    # Create a new image with the specified width, height, and background color
    image = Image.new("RGB", (width, height), background_color)
    return image
    



def load_claim(img_input, text_input, node_thres_input_level, edge_thres_input_level, vis_sim_thres_input_level):
    img_path = os.path.join(current_local_dir_path+'client_stuff','input_image.jpg')
    img_input.save(img_path)
    result = client.predict(
            img_claim=img_path,
            text_claim=text_input,
            node_thres_input_level=node_thres_input_level,
            edge_thres_input_level=edge_thres_input_level,
            vis_sim_thres_input_level=vis_sim_thres_input_level,
            api_name="/load_claim")

     
    return process_result_claims(result,'load_claim')

def get_best_evidence(t_sug):
    result2 = client.predict(
        e_t_sug=t_sug,
        api_name="/get_best_evidence"
    )
    return process_result_evidence(result2,'get_best_evidence')


def process_result_evidence(result2,mode):
    new_CDG0=Image.open(result2[0])
    new_VDG0=Image.open(result2[1])
    new_TDG0=Image.open(result2[2])
    new_dom0=           result2[3]
    xv_cat0 =           result2[4]
    xv_text0=           result2[5] 
    xvs0    =           result2[6]
    xv_img0 =Image.open(result2[7])
    xv_gs0  =           result2[8]
    xt_gs0  =           result2[9]
    
    new_CDG1=Image.open(result2[10])
    new_VDG1=Image.open(result2[11])
    new_TDG1=Image.open(result2[12])
    new_dom1=           result2[13]
    xv_cat1 =           result2[14]
    xv_text1=           result2[15] 
    xvs1    =           result2[16]
    xv_img1 =Image.open(result2[17])
    xv_gs1  =           result2[18]
    xt_gs1  =           result2[19]
    t_claim =           result2[20]
    v_claim = Image.open(result2[21])
    found_flag, current_index, t_sug ,old_t_sug,xt,xt_scr= result2[22:]
    return [new_CDG0,new_VDG0,new_TDG0,new_dom0,xv_cat0,xv_text0,xvs0,xv_img0,xv_gs0,xt_gs0, new_CDG1,new_VDG1,new_TDG1,new_dom1,xv_cat1,xv_text1,xvs1, xv_img1,xv_gs1,xt_gs1, t_claim, v_claim, found_flag, current_index, t_sug ,old_t_sug,xt,xt_scr]
    

def process_result_claims(result,mode):
    t_sug = result[10]
    old_t_sug = ''
    found_flag = ''
    xt = result[13]
    ci = result[14]
    #new_CDG, label, img_claim, text_claim,  
    return [t_sug, old_t_sug, found_flag, xt, ci]
    

def process_sample(jidx):
    node_thres_input_level    = 1
    edge_thres_input_level    = 1
    vis_sim_thres_input_level = 2
    tot_processed             = 0
    
    while(True):
          # Get image and text sample using remiss_sample
          text_input, img_input = remiss_sample(jidx)

          # Load the claim [t_sug, old_t_sug, found_flag, xt, ci])
          claim_result = load_claim(img_input, text_input, node_thres_input_level, edge_thres_input_level, vis_sim_thres_input_level)
          
          if 'no_text_evidence' in claim_result[3]:
              print('skipped as no text evidence')
              global sample_idx
              global split_type
              output_folder = 'remiss_data/outputs/'+split_type+'/'+str(sample_idx)
              os.makedirs(output_folder, exist_ok=True) 
              continue

          # Extract the suggested search term from the claim result
          t_sug = claim_result[0]
          # Get the best evidence using the suggested search term
          try:
            evidence_result = get_best_evidence(t_sug)
            [new_CDG0, new_VDG0, new_TDG0, new_dom0, xv_cat0, xv_text0, xvs0, xv_img0, xv_gs0, xt_gs0, new_CDG1, new_VDG1, new_TDG1, new_dom1, xv_cat1, xv_text1, xvs1, xv_img1, xv_gs1, xt_gs1, text_claim, img_claim, found_flag, ci, t_sug, old_t_sug,xt,xt_scr]=evidence_result
            save_outputs(new_CDG0, new_VDG0, new_TDG0, new_dom0, xv_cat0, xv_text0, xvs0, xv_img0, xv_gs0, xt, xt_scr, xt_gs0, new_CDG1, new_VDG1, new_TDG1, new_dom1, xv_cat1, xv_text1, xvs1, xv_img1, xv_gs1, xt_gs1, text_claim, img_claim, found_flag, ci, t_sug, old_t_sug)
            tot_processed+=1
          except:
            print('failed ')
          if tot_processed==30:
             break    
         
    return 0

########################################    
def final_postprocessing(jidx):
    #input api output
    #output decision
    jsonl_file      = jsonl_files[jidx]
    global split_type
    split_type      = splits[jidx]
    
    found_evd       = 0
    
    test_samples    = load_jsonl_file(jsonl_file)
    data_path       = 'remiss_data/remiss_data_complete/' + split_type 
    vl_dict         = np.load(data_path + '/remiss_filter_marked.npz', allow_pickle=True)
    vl_dict         = vl_dict[vl_dict.files[0]]
    vl_dict         = vl_dict.flat[0] 
    
    output_base_folder = 'remiss_data/outputs/'+  split_type 
    # List all folders in the base directory
    all_folders = [os.path.join(output_base_folder, d) for d in os.listdir(output_base_folder) if os.path.isdir(os.path.join(output_base_folder, d))]
    # Filter only non-empty folders
    non_empty_folders = [folder for folder in all_folders if os.listdir(folder)]
    # Process each non-empty folder
    #pdb.set_trace()
    for output_folder in non_empty_folders:
         try:
            res=load_from_folder(output_folder)
                       
            xvs0   =      res[6]
            xt_scr =      res[27]
            xt     =      res[26]    
            xv_gs0 =      res[8]
            xt_gs0 =      res[9]
            
            #pdb.set_trace()
            if xt_gs0> 0.33  :
                  found_evd+=1
            else:
               if xt=='no_text_evidence'  and ( (xv_gs0 > 0.33 and xvs0 > 0.8) or (xv_gs0 > 0.1 and xvs0 > 0.9)) :
                  found_evd+=1 
            
         except:
            print('x')
    frac_found=found_evd/len(non_empty_folders) 
    return frac_found   
         
########################################
# Gradio UI setup
web_ui=True
if web_ui==True:
  with gr.Blocks(theme=gr.themes.Default(spacing_size="sm", radius_size="none", text_size="lg")) as app:
    with gr.Row():
        gr.Markdown("Remiss Client v1")

    with gr.Row():
        with gr.Column():
            sload = gr.Button(value="1 Start")
            gr.Markdown("### CLAIM")
            img_input = gr.Image(type='pil', label="Input Image")
            text_input = gr.Textbox(label="Input Text")
            
            split_dropdown = gr.Dropdown(choices=splits, label="Select Split")
            load_sample_button = gr.Button(value="Load Sample")
            load_processed_sample_button = gr.Button(value="Load Processed Sample")
            
            
            #gt_pristine_button = gr.Button(value="GT Pristine 0")
            #gt_fake_button     = gr.Button(value="GT Fake     1")
            
																
			
            gr.Markdown("### Configuration")   
            node_thres_input_level=gr.Slider(minimum=0, maximum=5, step=1, label="Entity Similarity")
            edge_thres_input_level=gr.Slider(minimum=0, maximum=5, step=1, label="Action Similarity")
            vis_sim_thres_input_level=gr.Slider(minimum=0, maximum=5, step=1, label="Visual Similarity")
            t_sug = gr.Textbox(label="Next Suggested Search Term", interactive=True) 
           
																	
            gr.Markdown("### STATUS")
            old_t_sug = gr.Textbox(label="Generated by Search Term")
            found_flag = gr.Textbox(label="Found Status")
            gt_label   = gr.Textbox(label="GT Annotation")
            ci = gr.Textbox(label="Index")
            
        with gr.Column():
            run = gr.Button(value="2 Get Evidence")
            gr.Markdown("### TEXT EVIDENCE")   
            xt = gr.Textbox(label="XT from Reverse Search with Image")
            xt_scr   =  gr.Textbox(label=" XT Text Similarity score")
            with gr.Row(): 
                with gr.Column(): 
                    new_TDG0 = gr.Image(type='pil', label="Text Evidence Graph")
                    xt_gs0 = gr.Textbox(label="XT Graph score")
                    new_CDG0 = gr.Image(type='pil', label="Claim text Graph")
                    gr.Markdown("### VIS EVIDENCE 1")   
                    new_VDG0 = gr.Image(type='pil', label="Visual Evidence Graph")
                    xv_gs0 = gr.Textbox(label="XV Graph score")
                    xv_img0 = gr.Image(type='pil', label="Current Visual Evidence")
                    xvs0 = gr.Textbox(label="Visual Similarity Cumulative")
                    xv_cat0 = gr.Textbox(label="Top Channels of Image Match")
                    xv_text0 = gr.Textbox(label="Visual Evidence Caption")
                    new_dom0 = gr.Textbox(label="Visual Evidence domain")  
              
                with gr.Column(): 
                    new_TDG1 = gr.Image(type='pil', label="Text Evidence Graph")
                    xt_gs1 = gr.Textbox(label="XT Graph score")
                    new_CDG1 = gr.Image(type='pil', label="Claim text Graph")   

                    gr.Markdown("### VIS EVIDENCE 2")   
                    new_VDG1 = gr.Image(type='pil', label="Visual Evidence Graph")
                    xv_gs1 = gr.Textbox(label="XV Graph score")
                    xv_img1 = gr.Image(type='pil', label="Current Visual Evidence")
                    xvs1 = gr.Textbox(label="Visual Similarity Cumulative")
                    xv_cat1 = gr.Textbox(label="Top Channels of Image Match")
                    xv_text1 = gr.Textbox(label="Visual Evidence Caption")
                    new_dom1 = gr.Textbox(label="Visual Evidence domain")  
                    
   
    sload.click(load_claim, inputs=[img_input, text_input, node_thres_input_level, edge_thres_input_level, vis_sim_thres_input_level], 
                outputs=[t_sug, old_t_sug, found_flag, xt, ci])
       
    load_sample_button.click(lambda split: remiss_sample(split_to_index[split]), inputs=[split_dropdown], outputs=[text_input, img_input])
    
    load_processed_sample_button.click(lambda split: load_remiss_sample_already(split_to_index[split]), inputs=[split_dropdown], outputs=[new_CDG0, new_VDG0, new_TDG0, new_dom0, xv_cat0, xv_text0, xvs0, xv_img0, xv_gs0, xt_gs0, new_CDG1, new_VDG1, new_TDG1, new_dom1, xv_cat1, xv_text1, xvs1, xv_img1, xv_gs1, xt_gs1, text_input, img_input, found_flag, ci, t_sug, old_t_sug,xt,xt_scr,gt_label])

    
    run.click(get_best_evidence, inputs=[t_sug], outputs=[new_CDG0, new_VDG0, new_TDG0, new_dom0, xv_cat0, xv_text0, xvs0, xv_img0, xv_gs0, xt_gs0, new_CDG1, new_VDG1, new_TDG1, new_dom1, xv_cat1, xv_text1, xvs1, xv_img1, xv_gs1, xt_gs1, text_input, img_input, found_flag, ci, t_sug, old_t_sug,xt,xt_scr])
    
    save_button = gr.Button(value="Save Outputs")
    
    
    save_button.click(save_outputs, inputs=[new_CDG0, new_VDG0, new_TDG0, new_dom0, xv_cat0, xv_text0, xvs0, xv_img0, xv_gs0,xt,xt_scr,xt_gs0, new_CDG1, new_VDG1, new_TDG1, new_dom1, xv_cat1, xv_text1, xvs1, xv_img1, xv_gs1, xt_gs1, text_input, img_input, found_flag, ci, t_sug, old_t_sug],
                            outputs=[])
    
          
    #gt_pristine_button.click(user_input_labels_PRISTINE, inputs=[],  outputs=[])
    #gt_fake_button.click(    user_input_labels_FAKE, inputs=[],  outputs=[])                        
	
	
  app.launch()
  

  
#pdb.set_trace()
#final_postprocessing(0)
#process_sample(0)
#remiss_sample_already_stats(0)


	
						
														   
					   
						   
						   				  
						
'''
splits          = ['bcn19','gen19','gen21','mena_aggr','mena_ajud','openarms']
jidx            = 3

import numpy as np

split_type      = splits[jidx]
data_path       = 'remiss_data/remiss_data_complete/'+split_type 
vl_dict         = np.load(data_path+'/remiss_filter_marked.npz',allow_pickle=True)
vl_dict         = vl_dict[vl_dict.files[0]]
vl_dict         = vl_dict.flat[0]
vt_scores = [value['vt_score'] for value in vl_dict.values() if 'vt_score' in value]

# Calculate statistics
print('<')
print(split_type)
np.min(vt_scores)
np.max(vt_scores)
np.mean(vt_scores)
np.std(vt_scores)
sum(1 for value in vl_dict.values() if value.get('vt_score', 0) > 0.5)
print('>')


''' 
			 


								

											  
						
					
							   
 
			 
	
