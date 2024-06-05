from transformers import ViTFeatureExtractor, ViTForImageClassification
vit_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
#import face_recognition
import torch
import pdb
import traceback
from PIL import Image
#import spacy
import imghdr     
import imageio  
import cv2
import os
import glob
import shutil
from torchvision import models, transforms
import numpy as np
from scipy.optimize import linear_sum_assignment
from utils.config import dataset_name


#bert

from transformers import BertTokenizer, BertModel
if dataset_name=='remiss':
   tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')#('bert-base-uncased')
   bertmodel = BertModel.from_pretrained('bert-base-multilingual-uncased')#('bert-base-uncased')
else:
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')#('bert-base-uncased')
   bertmodel = BertModel.from_pretrained('bert-base-uncased')#('bert-base-uncased')


##<<---------------------------------------------------------------------------------
import timm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vitmodel = timm.create_model('vit_base_patch16_224', pretrained=True) 
vitmodel.eval().to(device)
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
vit_config = resolve_data_config({}, model=vitmodel)
transform_vit_sem = create_transform(**vit_config )

#------------------------------------------------------------------------------------------#
from utils import resnet_places
resnet_plc = resnet_places.PlacesCNN('resnet50')
resnet_plc.eval().to(device)
#..........................................................................................#
from facenet_pytorch import MTCNN, InceptionResnetV1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#mtcnn = MTCNN(margin=40, select_largest=True, keep_all=False, post_process=False, device=device)
mtcnn = MTCNN(margin=10, select_largest=False, keep_all=True, post_process=False, device=device)

resnet_face = InceptionResnetV1(pretrained='vggface2').eval().to(device)


################################

def encode_text(text, tokenizer, bertmodel):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bertmodel(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


'''def initialize_bert():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model'''
    
    
#blipcap
from transformers import BlipProcessor, BlipForConditionalGeneration

blipprocessor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blipmodel     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")


from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering

blip2processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip2model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-opt-2.7b").to("cuda")



def get_cap(v_path,prompt):
    raw_image=Image.open(v_path).convert('RGB')
    inputs = blipprocessor(raw_image,prompt, return_tensors="pt").to("cuda")
    out = blipmodel.generate(**inputs)
    gen_cap=blipprocessor.decode(out[0], skip_special_tokens=True)
    return gen_cap


def get_answer(v_path,prompt):
    raw_image=Image.open(v_path).convert('RGB')
    inputs = blip2processor(raw_image,prompt, return_tensors="pt").to("cuda")
    out = blip2model.generate(**inputs)
    gen_cap=blip2processor.decode(out[0], skip_special_tokens=True)
    return gen_cap


def get_exh_cap(v_path):
      gen_cap=[]
      sce_txt=''
      #prompts=['the place','the people','the subject',' ','it says:','exact text on the image  from top left to bottom right is :',' what is happening?']
      
      prompts=['the place','the people','the subject','<>',' what is happening?','exact text on the image  from top left to bottom right is :']

      
      pq=prompts.index('<>')
      for prompt in prompts[0:pq+1]:
         gen_cap.append(get_cap(v_path,prompt))
      
      gen_cap.append(get_answer(v_path,prompts[pq+1]))
      sce_txt=get_answer(v_path,prompts[pq+2])
       
      
      
      #pdb.set_trace()
      gen_cap              =' '.join(gen_cap)
      gen_cap_enc          = encode_text(gen_cap, tokenizer, bertmodel)
      sce_txt_enc          = encode_text(sce_txt, tokenizer, bertmodel)
     
      return gen_cap_enc.squeeze(0),gen_cap,sce_txt_enc.squeeze(0),sce_txt
      
          
def old_get_exh_cap(v_path):
      gen_cap=[]
      prompts=['the place','the people','the subject',' ','it says:','exact text on the image  from top left to bottom right is :',' what is happening?']
      pq=prompts.index(' ')
      for prompt in prompts[0:pq+1]:
         gen_cap.append(get_cap(v_path,prompt))
      for prompt in prompts[pq+1:]:
         gen_cap.append(get_answer(v_path,prompt)) 
      
      
      #pdb.set_trace()
      gen_cap              =' '.join(gen_cap)
      #tokenizer, bertmodel = initialize_bert()
      gen_cap_enc          = encode_text(gen_cap, tokenizer, bertmodel)
      return gen_cap_enc.squeeze(0),gen_cap
###


def get_face_embedding(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    
    if not face_locations:
        return None
    
    face_encoding = face_recognition.face_encodings(image, face_locations)[0]
    return face_encoding

def get_object_features(image_path):
    #image = Image.open(image_path)
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.mobilenet_v2(pretrained=True)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        input_batch = input_batch.to(device)
        output = model(input_batch)

    features = output[0]

    return features


def get_vit_features(image_path):
    # Load the image
    image = Image.open(image_path)

    # Define preprocessing steps
    preprocess = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
    ])

    # Preprocess the image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Make the prediction
    with torch.no_grad():
        input_batch = vit_feature_extractor(images=input_batch, return_tensors="pt")
        logits = vit_model(**input_batch).logits

    # Extract the features from the model output
    features = logits.squeeze()

    return features

# ... (rest of the code)


def get_place_features(image_path):
    image = Image.open(image_path)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(pretrained=True)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        input_batch = input_batch.to(device)
        output = model(input_batch)

    features = output[0]

    return features


def  aimread(img_path):
     gext=   img_path.split(".")[-1]
     pext=   imghdr.what(img_path)
     if pext in ['jpeg','jpg'] or pext==None:
              img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
              
     else:
         #print ('<< at '+img_path+' change'+ gext +'to'+ pext +' >>')
         corr_img_path=img_path.replace(gext,pext)
         if   os.path.isfile(corr_img_path)==False:
              shutil.copyfile(img_path,corr_img_path)
         if pext=='gif':
             gif = imageio.mimread(corr_img_path, memtest=False)
             imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in gif]    
             img=imgs[0]
         else:
             img = cv2.cvtColor(cv2.imread(corr_img_path), cv2.COLOR_BGR2RGB) 
         

     return img 




def get_faceemb(v_sem):
    success = 1
    if v_sem.size[0] > 1:
        faces, probs = mtcnn(v_sem, return_prob=True)
        if faces is not None and len(faces) > 0:
            # Select up to three faces
            faces = faces[:3]
            probs = probs[:3]
            success = 0

    if success == 0:
        faces_emb = resnet_face(faces.to(device))
    else:
        faces = np.array([[0, 0, 0, 0]])
        faces_emb = torch.zeros((1, 512)).to(device)

    return success, faces, faces_emb

def old_get_faceemb(v_sem):   ## taking only the largest face, not good, becacause of multple ppl
    success=1
    if v_sem.size[0]>1:
      faces, probs = mtcnn(v_sem, return_prob=True)
      faces_bbx, probs = mtcnn.detect(v_sem)
      if probs[0]!=None:
         success=0
    if success==0: 
         faces_emb=resnet_face(   faces.unsqueeze(0).to(device) )
    else:
         faces_bbx=np.array([[0,0,0,0]])
         faces_emb=torch.zeros((1,512)).to(device)    
        
    return success,faces_bbx,faces_emb  

def pre_img_feats(img_path):  
        raw_img=aimread(img_path)
        
        img                          = Image.fromarray(raw_img)
        success_fce,bbx_fce,v_fce    = get_faceemb(img)  
        transformed_img              = transform_vit_sem(img).unsqueeze(0).to(device)
        v_sem                        = vitmodel(  transformed_img  )   
        v_plc                        = resnet_plc( transformed_img, features='pool')   
        

        '''cd={}
        cd['f_flag'] =success
        cd['fce_bbx']=fce_bbx
        cd['v_fce']  =v_fce
        cd['v_sem']  =v_sem
        cd['v_plc']  =v_plc
        save_dir=PRC_DIR+ os.path.dirname(img_path)
        isExist = os.path.exists(save_dir)
        if not isExist:
               os.makedirs(save_dir)       
        save_path=PRC_DIR+img_path.replace(img_path.split(".")[-1],'npy')
        #np.save(save_path,cd) '''
        return v_sem,v_plc,v_fce,success_fce   

##----------------------------------------------------------------------------------------------->>


def calculate_similarity_matrix(face_features1, face_features2):
    similarity_matrix = np.zeros((len(face_features1), len(face_features2)))

    for i, features1 in enumerate(face_features1):
        for j, features2 in enumerate(face_features2):
            similarity_matrix[i, j] = torch.nn.functional.cosine_similarity( features1.unsqueeze(0), features2.unsqueeze(0) ).item()

    return similarity_matrix

def match_faces_hungarian(face_features1, face_features2):
    similarity_matrix = calculate_similarity_matrix(face_features1, face_features2)

    # Use the Hungarian algorithm to find the optimal assignment
    row_indices, col_indices = linear_sum_assignment(1-similarity_matrix)

    matching_pairs = [(face_features1[i], face_features2[j]) for i, j in zip(row_indices, col_indices)]
    total_similarity = similarity_matrix[row_indices, col_indices]#.sum()

    return matching_pairs, total_similarity


def calculate_similarity(face_features1, face_features2,sucf1, sucf2, object_features1, object_features2, place_features1, place_features2,vit_features1,vit_features2,gencap_features1,gencap_features2,scetxt_features1,scetxt_features2,suc_sct):
    
    #pdb.set_trace()
    
    wtfce=0.15
    wtobj=0.15
    wtvit=0.2
    wtplc=0.15
    wtcap=0.15
    wtsct=0.2

    if (sucf1!=0) or (sucf2!=0):
       #print('face <FAIL>')
       wtfce=0.0
       wtobj=0.2
       wtvit=0.2
       wtplc=0.2
       wtcap=0.15
       wtsct=0.25
       
    if (suc_sct!=0): 
       wtvit+=0.1
       wtplc+=0.1  
       wtcap+=0.05   
          
    dum,face_similarities=match_faces_hungarian(face_features1,   face_features2)
    face_similarities    =(1-sucf1)*(1-sucf2)*face_similarities
    face_similarity      =face_similarities.mean()
    object_similarity = torch.nn.functional.cosine_similarity(object_features1.unsqueeze(0), object_features2.unsqueeze(0)).item()
    vit_similarity    = torch.nn.functional.cosine_similarity(   vit_features1,    vit_features2).item()
    place_similarity  = torch.nn.functional.cosine_similarity( place_features1,  place_features2).item()
    gencap_similarity = torch.nn.functional.cosine_similarity(gencap_features1.unsqueeze(0), gencap_features2.unsqueeze(0)).item()
    scetxt_similarity = (1-suc_sct)*torch.nn.functional.cosine_similarity(scetxt_features1.unsqueeze(0), scetxt_features2.unsqueeze(0)).item()
    
    scores_list=[vit_similarity,place_similarity,face_similarity,object_similarity,gencap_similarity,scetxt_similarity,face_similarities]
    similarity_score = wtfce * (face_similarity) + wtobj * (object_similarity) + wtvit * (vit_similarity) + wtplc * (place_similarity) + wtcap * (gencap_similarity) +wtsct*scetxt_similarity
    ##<<
    topk=4
    vis_vpfocs=scores_list[0:6]
    vscr_vit,vscr_place,vscr_face,vscr_obj,vscr_cap,vscr_sct=vis_vpfocs
    score_cat=['vit','place','faces','objects','caption','scene_text']
    vis_cuml, score_topkcat = sum(sorted(vis_vpfocs)[-topk:]) / topk, [score_cat[i] for i in sorted(range(len(vis_vpfocs)), key=lambda i: vis_vpfocs[i], reverse=True)[:topk]]    
    similarity_score=vis_cuml
    ##
    return similarity_score,scores_list,score_topkcat

def get_similarity(img1_path, img2_path):
    
    
    
    '''face_encoding1 = get_face_embedding(img1_path)
    face_encoding2 = get_face_embedding(img2_path)

    place_features1 = get_place_features(img1_path)
    place_features2 = get_place_features(img2_path)

    vit_features1 = get_vit_features(img1_path)    
    vit_features2 = get_vit_features(img2_path)'''
    
    vit_features1, place_features1,face_features1,sucf1 =pre_img_feats(img1_path)
    vit_features2, place_features2,face_features2,sucf2 =pre_img_feats(img2_path)
    
    object_features1 = get_object_features(img1_path)
    object_features2 = get_object_features(img2_path)

    gencap_feat1,gencap_raw1,scetxt_feat1,scetxt_raw1  = get_exh_cap(img1_path)
    gencap_feat2,gencap_raw2,scetxt_feat2,scetxt_raw2  = get_exh_cap(img2_path)
    
    suc_sctext=0 
    if len(scetxt_raw2.replace(" ", "").replace("\n", "")) < 5 or  len(scetxt_raw1.replace(" ", "").replace("\n", "")) < 5:  
        print('scene text is bad, ignoring')
        suc_sctext=1


    similarity_score,scores_list,score_topkcat = calculate_similarity(face_features1, face_features2, sucf1, sucf2, object_features1, object_features2, place_features1, place_features2,vit_features1,vit_features2,gencap_feat1,gencap_feat2,scetxt_feat1,scetxt_feat2,suc_sctext)
   
    
    #if len(scetxt_raw2.replace(" ", "").replace("\n", "")) < 5 or  len(scetxt_raw1.replace(" ", "").replace("\n", "")) < 5:
    #       scores_list[5]=0
    #       score_topkcat.remove('scene_text')
    #pdb.set_trace()
    
    return similarity_score,scores_list,score_topkcat
 
    ###############>>>>>
    
##############################3    
def compare_texts(query_text, evidence_texts):
    #tokenizer, bertmodel = initialize_bert()
    
    query_text_tokens = tokenizer(query_text,max_length=512,truncation=True, return_tensors='pt')
    query_text_embeddings = bertmodel(**query_text_tokens).last_hidden_state.mean(dim=1)

    scores = []
    for evidence_text in evidence_texts:
        evidence_text_tokens = tokenizer(evidence_text, max_length=512,truncation=True,return_tensors='pt')
        evidence_text_embeddings = bertmodel(**evidence_text_tokens).last_hidden_state.mean(dim=1)
        
        score= torch.nn.functional.cosine_similarity(query_text_embeddings, evidence_text_embeddings).item()

        scores.append(score)

    return scores
def compare_images(query_image_path, evidence_image_paths):
    scores = []
    scores_list = []
    score_topkcat_lofl=[]
    for evidence_image_path in evidence_image_paths:
        try:
            score,score_list,score_topkcat=get_similarity(query_image_path, evidence_image_path)
            #print('>>>>>>>>')
            #print(score_list)
            #print(score)
            #print(evidence_image_path)
            #print('>>>>>>>>')
        except  Exception as e:
            print('image data issue: ')
            traceback.print_exc()
            print(f"An error occurred: {e}")
            print(evidence_image_path)
            print(query_image_path) 
            pdb.set_trace()
            score=0 
            score_list=[0,0,0,0,0,0,0]
            score_topkcat=['na','na']
        scores.append(score)
        scores_list.append(score_list)
        score_topkcat_lofl.append(score_topkcat)

    return scores,scores_list,score_topkcat_lofl
    
    
def compare_query_with_evidence(query_image_path, query_text, vis_evidence_caps, vis_evidence_paths,text_evidences):
    #pdb.set_trace()
    #print('visscore has scen text')
    
    image_scores,image_scores_list,score_topkcat_lofl = compare_images(query_image_path, vis_evidence_paths)
    image_cap_scores = compare_texts(query_text, vis_evidence_caps)
    text_scores      = compare_texts(query_text, text_evidences)
    return  image_scores,image_scores_list, score_topkcat_lofl, image_cap_scores, text_scores
    
    
    
    
    
    
    
def compare_query_with_vis_evidence(query_image_path, query_text, vis_evidence_caps, vis_evidence_paths):
    #pdb.set_trace()
    
    image_scores,image_scores_list,score_topkcat_lofl= compare_images(query_image_path, vis_evidence_paths)
    image_cap_scores                 = compare_texts(query_text, vis_evidence_caps)
    return  image_scores,image_scores_list,score_topkcat_lofl, image_cap_scores    
    
    
    
    
##########################33    
    
    
    
    
        
