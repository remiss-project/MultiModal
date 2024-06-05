
import json
import requests
from PIL import Image,ImageDraw
from io import BytesIO
from googletrans import Translator
import os
from tqdm import tqdm
import pdb
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
import matplotlib.pyplot as plt
from utils.common_utils import read_json_data,load_jsonl_file
from utils.rav_utils    import download_and_display_image
translator = Translator()
from uv_r.is_fake import uvtmodel
uvtmodel=uvtmodel('uv_r/RFmodel.joblib')


##vt<<
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
wt_emb_model = SentenceTransformer('bert-base-nli-mean-tokens')
from transformers import BlipProcessor, BlipForConditionalGeneration
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blipmodel     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")
##vt>>


from transformers import ViTImageProcessor, ViTForImageClassification
import requests
vitprocessor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
vitmodel = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')


def get_image_class(image):
     inputs = vitprocessor(images=image, return_tensors="pt")
     outputs = vitmodel(**inputs)
     logits = outputs.logits
     # model predicts one of the 1000 ImageNet classes
     predicted_class_idx = logits.argmax(-1).item()
     print(predicted_class_idx)
     print('pdclidx')
     #pc=vitmodel.config.id2label[predicted_class_idx]
     return predicted_class_idx





def get_histogram_top_k(org_label_dict,samplelist,k):
    
    ##create custom_label_dict with keys present in sample list and values i
    label_dict = {k: org_label_dict[k] for k in org_label_dict.keys() if k in samplelist}
    
    
    class_names = [vitmodel.config.id2label[label_dict[k]] for k in label_dict.keys()]
    
    #class_names = [vitmodel.config.id2label[label_dict[k]] for k in label_dict.keys() if label_dict[k] !=916]
   
    class_counts = Counter(class_names)
    
    # Get top 10 most frequent classes
    top_10_classes = class_counts.most_common(k)
    
    # Plot histogram for top 10 classes only
    plt.bar([c[0] for c in top_10_classes], [c[1] for c in top_10_classes])
    plt.xlabel('Class Names')
    plt.ylabel('Frequency')
    plt.title('Histogram of Top '+str(k)+' Most Frequent Classes')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('top_'+str(k)+'_histogram_nameme.png')  # Save the histogram as an image
    plt.close()  # Close the plot without displaying it



def ann_remiss_image(jidx):

 jsonl_files     = ['remiss_data/barcelona_2019_search-1.media.jsonl', 'remiss_data/generales_2019_search-1.media.jsonl', 'remiss_data/generalitat_2021_search-1.media.jsonl']
 splits           =['bcn19','gen19','gen21']

 jsonl_file = jsonl_files[jidx]
 split      = 'nf_'+splits[jidx]
 data       = load_jsonl_file(jsonl_file)
 savefile   = 'remiss_data/'+split+'/remiss_marked_img.npz'
 loadfile   = 'remiss_data/nf_bcn19/remiss_marked.npz'
 #savefile   = 'remiss_data/nf_bcn19/remiss_marked_annt_marked.npz'
 #loadfile   = 'remiss_data/nf_bcn19/annotation_labels.npz'
 
 vlmarked_dict       = np.load(loadfile,allow_pickle=True)
 vlmarked_dict       = vlmarked_dict[vlmarked_dict.files[0]]
 vlmarked_dict       = vlmarked_dict.flat[0]
 vlkeys              =list(vlmarked_dict.keys())   
 ann_dict   = {}
 skipped    = 0
 marked     = 0
 upperlimit = 20
 all_url    = []   
 #pdb.set_trace()
 #for current_index in tqdm(range(len(vlkeys))): #
 #   current_index  =vlkeys[current_index]

 for current_index in tqdm(range(len(data))): #
 

    item          = data[current_index]
    org_text      = item['text']
    
    ##
    #pdb.set_trace()
    img_path      ='remiss_data/'+split+'/claim_images/'+'claim_'+str(item['id'])+'_img.jpg'
    if not os.path.exists(img_path):
           print('no image file')
           continue
    img = Image.open(img_path)
    
    try:
        ann_dict[current_index]=get_image_class(img)
    except:
        print(current_index)
        print('image dim/format issue?')
    
    #print(ann_dict[current_index])    
        
 print(' annotation complete  for split : '+str(split))
 print(' marked :<'+str(marked)+'>  skipped: <'+str(skipped)+'> of total : <'+str(current_index))
 np.savez(savefile,ann_dict=ann_dict)
 
 pdb.set_trace() 
 return 0
 
 
 
def ann_remiss_uv(jidx):

 jsonl_files     = ['remiss_data/barcelona_2019_search-1.media.jsonl', 'remiss_data/generales_2019_search-1.media.jsonl', 'remiss_data/generalitat_2021_search-1.media.jsonl']
 splits           =['bcn19','gen19','gen21']

 jsonl_file = jsonl_files[jidx]
 split      = 'nf_'+splits[jidx]
 data       = load_jsonl_file(jsonl_file)
 savefile   = 'remiss_data/'+split+'/remiss_marked_uv45.npz'
 loadfile   = 'remiss_data/nf_bcn19/remiss_marked.npz'
 vlmarked_dict       = np.load(loadfile,allow_pickle=True)
 vlmarked_dict       = vlmarked_dict[vlmarked_dict.files[0]]
 vlmarked_dict       = vlmarked_dict.flat[0]
 vlkeys              =list(vlmarked_dict.keys())  
 
 ann_dict   = {}
 skipped    = 0
 marked     = 0
 upperlimit = 20
 all_url    = []   
 #pdb.set_trace()
 for current_index in tqdm(range(len(vlkeys))): #
    current_index  =vlkeys[current_index]
    item          = data[current_index]
    org_text      = item['text']
    
    ##
    #pdb.set_trace()
    #img_path      ='remiss_data/nf_bcn19/claim_images/'+'claim_'+str(item['id'])+'_img.jpg'
    #img = Image.open(img_path)
    #get_image_class(img)
        
    ##
    uv_fakeprob    = uvtmodel.predict([org_text])[0]
    print(uv_fakeprob)
    if uv_fakeprob>0.45:
    
                     ann_dict[current_index]=uv_fakeprob#uv_fakeprob
                     #print('<------------------ vtalign'+str(vt_align_score))
                     #print(en_text)
                     #print('                   ')
                     #print('                   ')
                     #print(gen_cap)
                     #print('------------------>') 
                     #print('>>>>>>>>>>>>>>>>>>>> <MARKED>')
                     
                     marked+=1

        
 print(' annotation complete  for split : '+str(split))
 print(' marked :<'+str(marked)+'>  skipped: <'+str(skipped)+'> of total : <'+str(current_index))
 np.savez(savefile,ann_dict=ann_dict)
  
 return 0
 
def ann_remiss_data(jidx):

 jsonl_files     = ['remiss_data/barcelona_2019_search-1.media.jsonl', 'remiss_data/generales_2019_search-1.media.jsonl', 'remiss_data/generalitat_2021_search-1.media.jsonl']
 splits           =['bcn19','gen19','gen21']

 jsonl_file = jsonl_files[jidx]
 split      = 'nf_'+splits[jidx]
 data       = load_jsonl_file(jsonl_file)
 ann_dict   = {}
 skipped    = 0
 marked     = 0
 upperlimit = 20
 all_url    = []   
 #pdb.set_trace()
 for current_index in tqdm(range(len(data))): #
    item          = data[current_index]
    org_text      = item['text']
    try:
       media         = item.get('media', [])
       img_path      ='remiss_data/'+split+'/claim_images/'+'claim_'+str(item['id'])+'_img.jpg'
       if media and media[0].get('type') == 'photo' and  media[0].get('url')!=False:
          marked+=1
    except:
        #pdb.set_trace()
        skipped+=1     
        
        
 print(' annotation complete  for split : '+str(split))
 print(' marked :<'+str(marked)+'>  skipped: <'+str(skipped)+'> of total : <'+str(current_index))
 pdb.set_trace() 
 return 0

def ann_remiss(jidx):

 jsonl_files     = ['remiss_data/barcelona_2019_search-1.media.jsonl', 'remiss_data/generales_2019_search-1.media.jsonl', 'remiss_data/generalitat_2021_search-1.media.jsonl']
 splits           =['bcn19','gen19','gen21']

 jsonl_file = jsonl_files[jidx]
 split      = 'nf_'+splits[jidx]
 data       = load_jsonl_file(jsonl_file)
 savefile   = 'remiss_data/'+split+'/remiss_marked_chk773.npz'
 
 loadfile   = 'remiss_data/nf_bcn19/remiss_marked.npz'
 vlmarked_dict       = np.load(loadfile,allow_pickle=True)
 vlmarked_dict       = vlmarked_dict[vlmarked_dict.files[0]]
 vlmarked_dict       = vlmarked_dict.flat[0]
 vlkeys              =list(vlmarked_dict.keys())  
 

 ann_dict   = {}
 skipped    = 0
 marked     = 0
 upperlimit = 20
 all_url    = []   
 #pdb.set_trace()
 #for current_index in tqdm(range(len(data))): #
 for current_index in tqdm(range(len(vlkeys))): #
    current_index  =vlkeys[current_index]
    item          = data[current_index]
    org_text      = item['text']
    try:
       media         = item.get('media', [])
       img_path      ='remiss_data/'+split+'/claim_images/'+'claim_'+str(item['id'])+'_img.jpg'
       if not os.path.exists(img_path):
           print('no file')
           pdb.set_trace()
       print('ck1-<>')    
       if media and media[0].get('type') == 'photo' and  media[0].get('url')!=False:
          url            = media[0].get('url')
          #uv_fakeprob    = uvtmodel.predict([org_text])[0]
          #print(uv_fakeprob)
          if (url not in all_url) :# and ( ( uv_fakeprob>= 0.5) ) :
               #pdb.set_trace()
               all_url.append(url)
               
               try:
                 ##image,img_path = download_and_display_image(url,img_path)
                 image = Image.open(img_path)
               except:
                 print('image load fs')
                 pdb.set_trace()   
               
               ##<blip
               print('ck2-<>')
               print('--<blip stff')
               en_text   = translator.translate(org_text, dest='en').text
               raw_image = image.convert('RGB')
               inputs    = processor(raw_image, return_tensors="pt").to("cuda")
               out       = blipmodel.generate(**inputs)
               gen_cap   = processor.decode(out[0], skip_special_tokens=True)
               vt_align_score= cosine_similarity([wt_emb_model.encode(en_text)],[wt_emb_model.encode( gen_cap)]).squeeze(0).item() 
               ##>
               print('<------------------ vtalign'+str(vt_align_score))
               print(en_text)
               print('                   ')
               print('                   ')
               print(gen_cap)
               print('------------------>') 
                     
               if vt_align_score>0.4:
                     ann_dict[current_index]=vt_align_score#uv_fakeprob
                     #print('>>>>>>>>>>>>>>>>>>>> <MARKED>')
                     
                     marked+=1
               else:
                   print('vt aling less??!!')
                   #pdb.set_trace()      
    except:
        pdb.set_trace()
        skipped+=1     
        
        
 print(' annotation complete  for split : '+str(split))
 print(' marked :<'+str(marked)+'>  skipped: <'+str(skipped)+'> of total : <'+str(current_index))
 np.savez(savefile,ann_dict=ann_dict)
  
 return 0
 
 
#for i in range(0,1):
if __name__ == "__main__":
  """ Main function to compute out-of-context detection accuracy"""
  ann_remiss_image(0)      
  pdb.set_trace()      
       
       
       
       
       
       
       
       
       
       
       
       
       
#marked :<15490>  skipped: <7709> of total : <300942
       
       
       
       
       
           
