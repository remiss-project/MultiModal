# counterview


docker run -e GOOGLE_APPLICATION_CREDENTIALS="application_default_credentials.json"  --gpus 'device=2' -it -v "/data/users/arka/rav/runtime_data:/server_runtime_data_docker" -p 7860:7860 gradio-app
