
import os
import requests
from io import BytesIO
import shutil
import json  

import pdb
import matplotlib.pyplot as plt
import pdb

from PIL import Image,ImageDraw
from utils.dir_ser_adaptive import search_and_save_one_query_dyn, getdatafromjson

from utils.config import save_folder_path,split_type,sub_split,startdate,dataset_name

def download_and_display_image(image_url, save_path):
    if not image_url.startswith("http://") and not image_url.startswith("https://"):
        return None, None  # Return None for invalid URLs
    response = requests.get(image_url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        img.save(save_path)
        return img, save_path
    else:
        return None, None
   
   
 
def load_jsonl_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = [json.loads(line) for line in lines]
    return data
           
 
def old_wgsearch(t_sug,current_index,k): 
    text_query=t_sug
    suc=0
    xvp=[]
    xvc=[]
    try :
        xvp,xvc,xvd=gsearch(text_query,current_index,k)
    except:
        suc=1
        
    return suc,xvp,xvc
    
    
               
def wgsearch(t_sug,current_index,k): 
    text_query=t_sug
    suc=0
    xvp=[]
    xvc=[]
    xvd=[]
    try :
        xvp,xvc,xvd=gsearch(text_query,current_index,k)
    except:
        suc=1
        
    return suc,xvp,xvc,xvd


    
def gsearch(text_query,current_index,k):
    folder_path = os.path.join(save_folder_path,split_type,'direct_search',sub_split,str(current_index ) +'-'+str(k))
    if os.path.exists(folder_path):
       shutil.rmtree(folder_path)
    jsonfile =False
    ###
    #print('folder deletio')
    #pdb.set_trace()
    ###
    
    try:
          text_query=text_query#+startdate
          print('_______________starting search for: ' + str(current_index ) +'--'+str(k) +' st: '+str(text_query)) 
          jsonfile = search_and_save_one_query_dyn(text_query, current_index,k)
          print('skipping start date should i use text_query+startdate')
    except:
          print('_______________search failed for: ' + str(current_index ))
 
    if jsonfile:
        print('_______________found evidences')
        vis_evidence_path, vis_evidence_caps,vis_doms = getdatafromjson(jsonfile)
        print(vis_evidence_path)
    else:
        vis_evidence_path = []
        vis_evidence_caps = []
        vis_doms          = []  
    
    ##<<dbg
    #print('folder done ?')
    #pdb.set_trace()    
    ###dbg>>
    return vis_evidence_path, vis_evidence_caps,vis_doms 
    
    
    
def  getdatafrom_idxk(current_index,k,xv_idx):
        try:
           jsonfile = os.path.join(save_folder_path,split_type,'direct_search',sub_split,str(current_index ) +'-'+str(k),'direct_annotation.json') 
           vis_evidence_path, vis_evidence_caps,vis_doms = getdatafromjson(jsonfile)
           xv         =  Image.open(vis_evidence_path[xv_idx])
           xv_cap     =  vis_evidence_caps[xv_idx]
        except:
           xv    =Image.open('/data/users/arka/rav/remiss_data/no-results.png')
           xv_cap='no_text_evidence'  
           
        return xv,xv_cap


def  getalldatafrom(current_index,k):
        try:
           jsonfile = os.path.join(save_folder_path,split_type,'direct_search',sub_split,str(current_index ) +'-'+str(k),'direct_annotation.json') 
           vis_evidence_path, vis_evidence_caps,vis_doms = getdatafromjson(jsonfile)
        except:
            print('rav utils get date from k failed')   
            #pdb.set_trace()  
        return vis_evidence_path, vis_evidence_caps   
    
def get_fig_aclm(cl_edg_count_list, qafrac_list, bin_pred_list, label_list):
    # Initialize lists to store computed values
    ek=10
    edge_counts           = [[] for _ in range(ek+1)]  # 7 elements for 0 to 6 edges, and 1 more for 7 or more edges
    accuracy              = [0] * (ek+1) #[0,0,0,0,0,0,0]  #[]
    false_positives       = [0] * (ek+1) #[0,0,0,0,0,0,0]  #[]
    false_negatives       = [0] * (ek+1) #[0,0,0,0,0,0,0]  #[]
    qafrac_avg            = [0] * (ek+1) #[0,0,0,0,0,0,0]
    qafrac_fp             = [[] for _ in range(ek+1)]  # List to store QA fractions for false positives
    qafrac_fn             = [[] for _ in range(ek+1)]  # List to store QA fractions for false negatives
    sample_count_per_edge = [0] * (ek+1) #  # Number of samples for each claim edge count

    # Iterate through the data
    for i in range(len(cl_edg_count_list)):
        num_edges = min(cl_edg_count_list[i], ek)  # Maximum of 6 edges, then 7 or more
        edge_counts[num_edges].append(i)
        sample_count_per_edge[num_edges] += 1
    
    # Compute accuracy, false positives, and true negatives for each set
    for i, edge_set in enumerate(edge_counts):
        if not edge_set:
            continue
    
        true_positive  = 0
        false_positive = 0
        true_negative  = 0
        false_negative = 0
        qafrac_sum     = 0
        num_edges      = min(i, ek)
        
        for sample_idx in edge_set:
            if bin_pred_list[sample_idx] == label_list[sample_idx]:  # Correctly classified
                if bin_pred_list[sample_idx] == 0:  # True negative
                    true_negative += 1
                else:  # True positive
                    true_positive += 1
            else:  # Misclassified
                if bin_pred_list[sample_idx] == 0:  # False negative
                    false_negative += 1
                    qafrac_fn[num_edges].append(qafrac_list[sample_idx]) 
                else:  # False positive
                    false_positive += 1
                    qafrac_fp[num_edges].append(qafrac_list[sample_idx])
           
            # Accumulate qafrac
            qafrac_sum += qafrac_list[sample_idx]
                
        # Compute accuracy
        total_samples = len(edge_set)
        acc = (true_positive + true_negative) / total_samples
        accuracy[num_edges] = acc
    
        # Store false positives and true negatives
        false_positives[num_edges] = (false_positive / total_samples)
        false_negatives[num_edges] = (false_negative / total_samples)
        qafrac_avg[num_edges] = (qafrac_sum / total_samples)

    # Plotting
    plt.figure(figsize=(10, ek))

    plt.scatter(range(len(accuracy)), accuracy, marker='o', c='blue', alpha=0.5, label='Accuracy')
    plt.scatter(range(len(false_positives)), false_positives, marker='^', c='green', alpha=0.5, label='False Positives (True News -> we mark as as Fake)')
    plt.scatter(range(len(false_negatives)), false_negatives, marker='s', c='red', alpha=0.5, label='False Negatives (Fake News -> we mark as True)')

    # Adding small bars for average fraction of edges verified for false positives
    #for i, qafrac_fp_list in enumerate(qafrac_fp):
    #    avg_qafrac_fp = sum(qafrac_fp_list) / len(qafrac_fp_list) if qafrac_fp_list else 0
    #    plt.bar(i - 0.2, avg_qafrac_fp, color='gray', alpha=0.5, width=0.2, align='center' ,  label='Avg. Fraction of Edges Verified (False Positives)')
    
    # Adding small bars for average fraction of edges verified for false negatives
    #for i, qafrac_fn_list in enumerate(qafrac_fn):
    #    avg_qafrac_fn = sum(qafrac_fn_list) / len(qafrac_fn_list) if qafrac_fn_list else 0
    #    plt.bar(i + 0.2, avg_qafrac_fn, color='orange', alpha=0.5, width=0.2, align='center', label='Avg. Fraction of Edges Verified (False Negatives)')




    # Plot samples per edge normalized by total samples
    total_samples = sum(sample_count_per_edge)
    plt.plot([count / total_samples for count in sample_count_per_edge], marker='x', c='purple', linestyle='-', label='#Samples with this Edge count (Normalized)')

    plt.xlabel('Number of Edges in Claim')
    plt.ylabel('Percentage')
    plt.title('Accuracy, False Positives, False Negatives, and  #Samples with Edge count (Normalized) vs Number of Edges in Claim')
    plt.xticks(range(len(accuracy)), range(len(accuracy)))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('accuracy_vs_edge_count_wt'+dataset_name+'.png')
    plt.close()  # Close the plot to free memory

    return sample_count_per_edge

def old_get_fig_aclm(cl_edg_count_list, qafrac_list, bin_pred_list, label_list):
    # Initialize lists to store computed values
    
    edge_counts     = [[] for _ in range(7)]  # 7 elements for 0 to 6 edges, and 1 more for 7 or more edges
    accuracy        = [0,0,0,0,0,0,0]  #[]
    false_positives = [0,0,0,0,0,0,0]  #[]
    false_negatives = [0,0,0,0,0,0,0]  #[]
    qafrac_avg      = [0,0,0,0,0,0,0]
    qafrac_fp       = [[] for _ in range(7)]  # List to store QA fractions for false positives
    qafrac_fn       = [[] for _ in range(7)]  # List to store QA fractions for false negatives
    sample_count_per_edge = [0,0,0,0,0,0,0]  # Number of samples for each claim edge count

    # Iterate through the data
    for i in range(len(cl_edg_count_list)):
        num_edges = min(cl_edg_count_list[i], 6)  # Maximum of 6 edges, then 7 or more
        edge_counts[num_edges].append(i)
        sample_count_per_edge[num_edges] += 1
    
    # Compute accuracy, false positives, and true negatives for each set
    for i, edge_set in enumerate(edge_counts):
        if not edge_set:
            continue
    
        true_positive  = 0
        false_positive = 0
        true_negative  = 0
        false_negative = 0
        qafrac_sum     = 0
        num_edges      = min(i, 6)
        
        for sample_idx in edge_set:
            if bin_pred_list[sample_idx] == label_list[sample_idx]:  # Correctly classified
                if bin_pred_list[sample_idx] == 0:  # True negative
                    true_negative += 1
                else:  # True positive
                    true_positive += 1
            else:  # Misclassified
                if bin_pred_list[sample_idx] == 0:  # False negative
                    false_negative += 1
                    qafrac_fn[num_edges].append(qafrac_list[sample_idx]) 
                else:  # False positive
                    false_positive += 1
                    qafrac_fp[num_edges].append(qafrac_list[sample_idx])
           
            # Accumulate qafrac
            qafrac_sum += qafrac_list[sample_idx]
                
        # Compute accuracy
        total_samples = len(edge_set)
        acc = (true_positive + true_negative) / total_samples
        accuracy[num_edges] = acc
    
        # Store false positives and true negatives
        false_positives[num_edges] = (false_positive / total_samples)
        false_negatives[num_edges] = (false_negative / total_samples)
        qafrac_avg[num_edges] = (qafrac_sum / total_samples)

    # Plotting
    plt.figure(figsize=(10, 6))

    plt.scatter(range(len(accuracy)), accuracy, marker='o', c='blue', alpha=0.5, label='Accuracy')
    plt.scatter(range(len(false_positives)), false_positives, marker='^', c='green', alpha=0.5, label='False Positives (True -> unverified)')
    plt.scatter(range(len(false_negatives)), false_negatives, marker='s', c='red', alpha=0.5, label='False Negatives (Fake -> verified)')

    # Adding small bars for average fraction of edges verified for false positives
    for i, qafrac_fp_list in enumerate(qafrac_fp):
        avg_qafrac_fp = sum(qafrac_fp_list) / len(qafrac_fp_list) if qafrac_fp_list else 0
        plt.bar(i - 0.2, avg_qafrac_fp, color='gray', alpha=0.5, width=0.2, align='center')
    
    # Adding small bars for average fraction of edges verified for false negatives
    for i, qafrac_fn_list in enumerate(qafrac_fn):
        avg_qafrac_fn = sum(qafrac_fn_list) / len(qafrac_fn_list) if qafrac_fn_list else 0
        plt.bar(i + 0.2, avg_qafrac_fn, color='orange', alpha=0.5, width=0.2, align='center')

    '''# Plot empty bars for cases where there are no false positives or false negatives
    for i in range(7):
        if not qafrac_fp[i]:
            plt.bar(i - 0.2, 0, color='none', edgecolor='gray', alpha=0.5, width=0.2, align='center')
        if not qafrac_fn[i]:
            plt.bar(i + 0.2, 0, color='none', edgecolor='orange', alpha=0.5, width=0.2, align='center')'''


    # Plot samples per edge normalized by total samples
    total_samples = sum(sample_count_per_edge)
    plt.plot([count / total_samples for count in sample_count_per_edge], marker='x', c='purple', linestyle='-', label='Samples per Edge (Normalized)')

    plt.xlabel('Number of Edges in Claim')
    plt.ylabel('Percentage')
    plt.title('Accuracy, False Positives, False Negatives, and Samples per Edge (Normalized) vs Number of Edges in Claim')
    plt.xticks(range(len(accuracy)), range(len(accuracy)))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('accuracy_vs_edge_count_wt.png')
    plt.close()  # Close the plot to free memory

    return sample_count_per_edge

