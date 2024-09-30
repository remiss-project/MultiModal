#!/usr/bin/env python
# coding: utf-8
import requests
import json
import pdb
import time


key="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" 

max_limit=19000
def count_tokens(text):
    # Split the text into tokens (words or characters)
    tokens = text.split()
    # Count the number of tokens
    return len(tokens)

def get_response_ch_api_wrap(user_input):
        if     count_tokens(user_input) <  max_limit:
               res   = get_response_ch_api(user_input)
        else:
               combined_text=user_input
               tokens       = combined_text.split()   
               print ('<< TOKEN OVERLOAD>>')
               combined_text= ' '.join(tokens[:max_limit])
               res          = get_response_ch_api(combined_text)  
        return res
        
        
        
def get_response_ch_api(user_input):
  api_key = key

  print('LIVE RUN, DINERO DINERO')
  
  # The API endpoint URL
  api_url = "https://api.openai.com/v1/chat/completions"
  # Input message to ChatGPT
  # Specify the model
  model = "gpt-3.5-turbo"#"gpt-3.5-turbo-1106"#
  # Send a POST request to the API with the model parameter
  response = requests.post(api_url, headers={"Authorization": f"Bearer {api_key}"}, json={"messages": [{"role": "user", "content": user_input}], "model": model})

  # Parse the JSON response
  response_data = json.loads(response.text)


  while 'error' in response_data:
      #pdb.set_trace()
      print(response_data['error'])
      print('waiting for 20s, for Ratelimit')
      time.sleep(20)
      response = requests.post(api_url, headers={"Authorization": f"Bearer {api_key}"}, json={"messages": [{"role": "user", "content": user_input}], "model": model})
      response_data = json.loads(response.text)

  model_reply = response_data["choices"][0]["message"]["content"]

  #print(model_reply)
  return model_reply
