"""
    Utility file consisting of common functions and variables used during training and evaluation
"""

import json
import os
#import cv2

def load_jsonl_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = [json.loads(line) for line in lines]
    return data


def read_json_data(file_name):
    """
        Utility function to read data from json file

        Args:
            file_name (str): Path to json file to be read

        Returns:
            article_list (List<dict>): List of dict that contains metadata for each item
    """
    with open(file_name) as f:
        article_list = [json.loads(line) for line in f]
        return article_list



