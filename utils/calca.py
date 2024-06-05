import pdb
import numpy as np
import pdb


def calculate_confusion_matrix(ground_truth_labels, predicted_labels):
    # Initialize counters
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(ground_truth_labels)):
        if ground_truth_labels[i] == 0:  # Positive class
            if predicted_labels[i] == 0:
                TP += 1
            else:
                FN += 1
        else:  # Negative class
            if predicted_labels[i] == 1:
                TN += 1
            else:
                FP += 1
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    # Calculate fractions
    total_positives = TP + FN
    total_negatives = TN + FP

    if total_positives > 0:
        TP_fraction = TP / total_positives
        FN_fraction = FN / total_positives
    else:
        TP_fraction = 0
        FN_fraction = 0

    if total_negatives > 0:
        FP_fraction = FP / total_negatives
        TN_fraction = TN / total_negatives
    else:
        FP_fraction = 0
        TN_fraction = 0

    return TP_fraction, FP_fraction, TN_fraction, FN_fraction, accuracy    
    
def extract_predictions(all_res):
    xt_predictions = []
    sim_txt_predictions = []
    gmatch_txt_predictions = []
    
    sim_vxv_predictions = []
    imatch_vxv_predictions = []
    gmatch_txvt_predictions = []
    imatch_cat_predictions = []
    rav_predictions = []
    #all_res[ransamidx]={'xt':xt,'sim_txt':abl_sim_txt,'gmatch_txt':abl_gm_txt,'sim_vxv':abl_sim_vxv,'imatch_vxv':abl_im_vxv, 'gmatch_txvt':abl_gm_txvt, 'imatch_cat': score_cat_indices ,'rav':bin_pred}
    
    
    for key in all_res:
        prediction = all_res[key]
        xt_predictions.append(1-int(prediction['xt']))   ### true means present  -> 0 false ->1
        sim_txt_predictions.append(  prediction['sim_txt'])
        gmatch_txt_predictions.append( prediction['gmatch_txt'])
        sim_vxv_predictions.append(  prediction['sim_vxv'])
        imatch_vxv_predictions.append( prediction['imatch_vxv']) 
        gmatch_txvt_predictions.append(prediction['gmatch_txvt']) 
        imatch_cat_predictions.append(prediction['imatch_cat'])
        rav_predictions.append(prediction['rav'])

    return {
        'xt_predictions'         : xt_predictions,
        'sim_txt_predictions'    : sim_txt_predictions,
        'gmatch_txt_predictions' : gmatch_txt_predictions,
        'sim_vxv_predictions'    : sim_vxv_predictions,
        'imatch_vxv_predictions' : imatch_vxv_predictions,
        'gmatch_txvt_predictions': gmatch_txvt_predictions,
        'imatch_cat_predictions' : imatch_cat_predictions,
        'rav_predictions'        : rav_predictions
    }    
##############################################3
    
    
marked_list_labels=np.load('remiss_data/bcn19/annotation_labels_human_chimser.npz',allow_pickle=True)
marked_list_labels=marked_list_labels[marked_list_labels.files[0]]
marked_list_labels=marked_list_labels.flat[0]
marked_list_keys=list(marked_list_labels.keys())
marked_list_keys=[int(x) for x in marked_list_keys]


all_res  =np.load('remiss_data/bcn19/ablation.npz',allow_pickle=True)
#all_res  =np.load('newsclip_data/ablation.npz',allow_pickle=True)


all_res  =all_res[all_res.files[0]]
all_res  =all_res.flat[0]
all_res_keys  =list( all_res.keys())



#all_res_labels = [1 if key in marked_list_keys else 0 for key in all_res_keys]
all_res_labels = [1 if i % 2 == 0 else 0 for i in range(len(all_res_keys))]




num_fakes= sum(all_res_labels)
num_pris = len(all_res_keys) -num_fakes



predictions = extract_predictions(all_res)





pdb.set_trace()
for key in ('xt_predictions','sim_txt_predictions','gmatch_txt_predictions','sim_vxv_predictions','imatch_vxv_predictions','gmatch_txvt_predictions','rav_predictions'):
      print(key)
      TP, FP, TN, FN, Acc = calculate_confusion_matrix(all_res_labels,predictions[key])
      print("TP, FP, TN, FN, Acc:", TP, FP, TN, FN,Acc)
             



pdb.set_trace()

sim_t_fake = 0
sim_v_fake = 0
gmatch_fake = 0
imatch_fake = 0
imatchwgm_fake = 0
rav_fake = 0

# Initialize counters for true samples
sim_t_pris = 0
sim_v_pris = 0
gmatch_pris = 0
imatch_pris = 0
imatchwgm_pris = 0
rav_pris = 0
fakesampleswith_xt=0
prissampleswith_xt=0


#img_cat_votes=[]#todo
#all_res[ransamidx]={'xt':xt,'sim_t':abl_sim_txt,'gmatch':abl_gm,'sim_v':abl_sim_vis,'imatch':abl_im, 'imatch_cat': score_cat_indices ,'rav':bin_pred}
#all_res[ransamidx]={'xt':xt,'sim_t':abl_sim_txt,'gmatch':abl_gm,'sim_v':abl_sim_vis,'imatch':abl_im, 'imatchwgm':abl_imgm, 'imatch_cat': score_cat_indices ,'rav':bin_pred}
#all_res[ransamidx]={'xt':xt,'sim_txt':abl_sim_txt,'gmatch_txt':abl_gm_txt,'sim_v_vxv':abl_sim_vxv,'imatch_vxv':abl_im_vxv, 'gmatch_txvt':abl_gm_txvt, 'imatch_cat': score_cat_indices ,'rav':bin_pred}
   

for k in all_res_keys:
   if k in marked_list_keys: 
      if (all_res[k]['xt']==True):
         fakesampleswith_xt+=1
      if all_res[k]['sim_t']==1  and (all_res[k]['xt']==True):
         sim_t_fake+=1
      if all_res[k]['sim_v']==1:
         sim_v_fake+=1
      if all_res[k]['gmatch']==1 and (all_res[k]['xt']==True):
         gmatch_fake+=1
      if all_res[k]['imatch']==1:
         imatch_fake+=1
      if all_res[k]['imatchwgm']==1:
         imatchwgm_fake+=1   
      if all_res[k]['rav']==1:
         rav_fake+=1
   else:
        if (all_res[k]['xt']==True):
               prissampleswith_xt+=1  
        if all_res[k]['sim_t']   == 0 and (all_res[k]['xt']==True) :
               sim_t_pris += 1
        if all_res[k]['sim_v']   == 0:
               sim_v_pris += 1
        if all_res[k]['gmatch']  == 0 and (all_res[k]['xt']==True) :
                gmatch_pris += 1
        if all_res[k]['imatch']  == 0:
                imatch_pris += 1
        if all_res[k]['imatchwgm']  == 0:
                imatchwgm_pris += 1                
        if all_res[k]['rav']     == 0:
                rav_pris += 1
   

#pdb.set_trace()
# Print results for fake samples





# Print results for true samples
print("Number of true samples (sim_t=0):", sim_t_pris/prissampleswith_xt )
print("Number of fake samples (sim_t=1):", sim_t_fake/fakesampleswith_xt)
print("tot (sim_t):", (sim_t_pris+sim_t_fake)/(prissampleswith_xt+fakesampleswith_xt) )

print("Number of true samples (sim_t=0):", sim_t_pris/num_pris )
print("Number of fake samples (sim_t=1):", sim_t_fake/num_fakes)
print("tot (sim_t):", (sim_t_pris+sim_t_fake)/len(all_res_keys) )


print("Number of true samples (sim_v=0):", sim_v_pris/num_pris )
print("Number of fake samples (sim_v=1):", sim_v_fake/num_fakes)
print("tot (sim_v):", (sim_v_pris+sim_v_fake)/len(all_res_keys) )


print("Number of true samples (gmatch=0):", gmatch_pris/prissampleswith_xt )
print("Number of fake samples (gmatch=1):", gmatch_fake/fakesampleswith_xt)
print("tot (gmatch):", (gmatch_pris+gmatch_fake)/(prissampleswith_xt+fakesampleswith_xt)) 

print("Number of true samples (gmatch=0):", gmatch_pris/num_pris )
print("Number of fake samples (gmatch=1):", gmatch_fake/num_fakes)
print("tot (gmatch):", (gmatch_pris+gmatch_fake)/len(all_res_keys)) 



print("Number of true samples (imatch=0):", imatch_pris/num_pris )
print("Number of fake samples (imatch=1):", imatch_fake/num_fakes)
print("tot (imatch):", (imatch_pris+imatch_fake)/len(all_res_keys) )


print("Number of true samples (imatch=0):", imatchwgm_pris/num_pris )
print("Number of fake samples (imatch=1):", imatchwgm_fake/num_fakes)
print("tot (imatch):", (imatchwgm_pris+imatchwgm_fake)/len(all_res_keys) )

print("Number of true samples (rav=0):", rav_pris/num_pris )
print("Number of fake samples (rav=1):", rav_fake/num_fakes)
print("tot (rav):", (rav_pris+rav_fake)/len(all_res_keys) )

pdb.set_trace()















