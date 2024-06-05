from transformers import AutoModelForCausalLM, AutoTokenizer
import pdb
import networkx as nx
import os
import pdb
import matplotlib.pyplot as plt
import numpy as np
import io
import glob
import shutil
from PIL import Image, ImageDraw
import ast
device = "cuda"

model          = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer      = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")


def core_mistral(messages):
     print('<mistral run>')
     encodeds = tokenizer.apply_chat_template(messages, tokenize=True,return_tensors="pt")
     model_inputs = encodeds.to(device)
     model.to(device)
     generated_ids = model.generate(model_inputs, max_new_tokens=4096, do_sample=True,temperature=0.001)
     decoded = tokenizer.batch_decode(generated_ids)
     output=decoded[0].split('[/INST]')[1][0:-4]
     return output


def get_response_mistral_x(msg):
    messages = [{"role": "user", "content": msg }] 
    output   = core_mistral(messages)
    return output





    
    

       
