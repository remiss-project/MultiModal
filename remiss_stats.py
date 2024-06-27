import json
import requests
from PIL import Image,ImageDraw
from io import BytesIO
import os
from tqdm import tqdm
import pdb
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
import matplotlib.pyplot as plt
from utils.common_utils import read_json_data,load_jsonl_file
from transformers import ViTImageProcessor, ViTForImageClassification
vitprocessor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
vitmodel = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

jidx=5
k=10
jsonl_files     = ['remiss_data/remiss_data_complete/barcelona_2019_search-1.media.jsonl', 'remiss_data/remiss_data_complete/generales_2019_search-1.media.jsonl', 'remiss_data/remiss_data_complete/generalitat_2021_search-1.media.jsonl', 'remiss_data/remiss_data_complete/MENA_Agressions.media.jsonl', 'remiss_data/remiss_data_complete/MENA_Ajudes.media.jsonl', 'remiss_data/remiss_data_complete/Openarms.media.jsonl']
splits          = ['bcn19','gen19','gen21','mena_aggr','mena_ajud','openarms']
 
jsonl_file = jsonl_files[jidx]
split      = splits[jidx]
data       = load_jsonl_file(jsonl_file)
npfile     = 'remiss_data/remiss_data_complete/'+split+'/remiss_filter_marked.npz'
vl_dict    = np.load(npfile, allow_pickle=True)
vl_dict    = vl_dict[vl_dict.files[0]]
vl_dict    = vl_dict.flat[0] 
class_names = [vitmodel.config.id2label[vl_dict [k]['img_class']] for k in vl_dict .keys() if vl_dict [k]['img_class'] !=916]
class_counts = Counter(class_names)
    
# Get top 10 most frequent classes
top_10_classes = class_counts.most_common(10)
    
# Plot histogram for top 10 classes only
plt.bar([c[0] for c in top_10_classes], [c[1] for c in top_10_classes])
plt.xlabel('Class Names')
plt.ylabel('Frequency')
plt.title('Histogram of Top '+str(k)+' Most Frequent Classes')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top_'+str(k)+'_histogram_nameme'+str(jidx)+'_.png')  # Save the histogram as an image
plt.close()  # Close the plot without displaying it
print(len(vl_dict))
print(len(data))





