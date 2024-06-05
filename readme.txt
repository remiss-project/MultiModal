docker load -i gradio-app.tar



docker run --gpus '"device=2"' -it -v "/data2/users/arka/rav/runtime_data:/data2/users/arka/rav/runtime_data" -p 7860:7860  gradio-app

docker volume create runtime_eng_vol

docker run --gpus '"device=2"' -it -v runtime_eng_vol:/data2/users/arka/rav/runtime_data -p 7860:7860 gradio-app





docker run --gpus '"device=2"' -it \
  -v "/data2/users/arka/rav/runtime_data:/data2/users/arka/rav/runtime_data" \
  -p 7860:7860 \
  --user 2178:2100 \
  gradio-app






docker build -t gradio-app .

docker save -o en-gradio-app.tar gradio-app

##############################################
##############################################
mkdir -p runtime_data/live_run/claim_images \
         runtime_data/live_run/direct_search \
         runtime_data/res_chatgpt
tar -xvf upload.tar -C runtime_data/ 

#######################
docker load -i en-gradio-app.tar
docker run --gpus '"device=2"' -it -v "/data2/users/arka/rav/runtime_data:/data2/users/arka/rav/runtime_data" -p 7860:7860  gradio-app

 


#####
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"

##cloud vision docker
gcloud init
gcloud services enable vision.googleapis.com
gcloud projects add-iam-policy-binding aerobic-cosmos-425308-n8 --member="user:xadcx19@gmail.com" --role='roles/editor'
gcloud auth application-default login

cp ~/.config/gcloud/application_default_credentials.json   /data2/users/arka/rav/application_default_credentials.json


docker run -e GOOGLE_APPLICATION_CREDENTIALS="application_default_credentials.json" \
--mount type=bind,source=${HOME}/.config/gcloud,target=/app/.config/gcloud  \
--gpus 'device=5' \
-it -v "/data2/users/arka/rav/runtime_data:/data2/users/arka/rav/runtime_data" \
-p 7860:7860 \
gradio-app




