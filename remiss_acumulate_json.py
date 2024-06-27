import os
import json
import pdb
import numpy as np

# Define paths
DATA_DIR = '/data/users/arka/rav/outputs/'
labelnames=['PRISTINE','FAKE']

user_labelled=np.load('/data/users/arka/rav/outputs/user_input_labels.npz',allow_pickle=True)
user_labelled=user_labelled[user_labelled.files[0]]
user_labelled=user_labelled.flat[0]

global combined_dict
combined_metadata=[]


# Function to combine jdict.json files into a single JSON file
def combine_json_files(data_dir, res_folder):
    global combined_dict
    output_folders = [os.path.join(data_dir, res_folder, folder) for folder in os.listdir(os.path.join(data_dir, res_folder))]
    output_folders = [folder for folder in output_folders if os.listdir(folder)]

    for folder in output_folders:
        json_file_path = os.path.join(folder, 'data.json')
        if os.path.isfile(json_file_path):
            try:
               with open(json_file_path, 'r') as f:
                 jdict = json.load(f)
                 
                 ##
                 key= str(folder.split('/')[-2]) + '< >' + str(folder.split('/')[-1])
                
                 
                 #pdb.set_trace()
                 found_flag= 1
                 exp=''
                 xt   =jdict['text_evidence']
                 xt_gs=float( jdict['text_evidence_graph_similarity_score']   )
                 xv_gs=float( jdict['visual_evidence_graph_similarity_score'] )
                 xv_s =float( jdict['visual_evidence_similarity_score']       )
                 if xt_gs > 0.3  :
                      found_flag = 0
                      exp +='XT found'
                 if xt=='no_text_evidence'  and ( (  xv_gs > 0.33 and xv_s > 0.8) or (xv_gs > 0.1 and xv_s > 0.9)) :
                      found_flag = 0
                      exp +='XV found' 
                 
                 jdict['results']={'predicted_label':labelnames[found_flag],"actual_label": labelnames[user_labelled[key]],"visual_similarity_score":xv_s,"explanations":exp}
                 #pdb.set_trace()
                 ##
                 
                 
                 combined_metadata.append(jdict)
            except:
                print(json_file_path) 





splits          = ['bcn19','mena_aggr','mena_ajud','openarms']

for res_folder in splits:
     combine_json_files(DATA_DIR, res_folder)


pdb.set_trace()
print('done')
pdb.set_trace()

with open('combined_metadata.json', 'w') as file:
    json.dump(combined_metadata, file, indent=4)



