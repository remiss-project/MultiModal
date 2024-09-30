# install requirements
pip install -r requirements.txt


# create and enable google cloud vision API , no changes in code required
https://cloud.google.com/vision/docs/setup

# create and set openai API, update in chtgpt.py
key="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" 

# create and set Programmable search engine id  and api in config.py
https://programmablesearchengine.google.com/about/
dir_ser_api     ='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX' 
gse_cid         ='XXXXXXXXXXXXXXXXXXXXXXX'

# run server
python chimser_rtl.py

