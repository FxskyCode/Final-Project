import random
import json
import webbrowser as wb
import subprocess
import os
import tools.sql_queries as sql
import sqlalchemy as alch


import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize


#un_list =["open google", "5 most valuable companies"]
fun_list = {"open google": "google()","open youtube": "youtube()","open monkeytype": "monkeytype()","send email":"sendemail()","search":"search_google()", "cleansing":"cleansing()","visuals":"visuals()", "sql top 5 valuation companies":"sql.get_top_valuation()"}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Iron"

def get_response(msg):
    if "search" in msg:
        search_google(msg)
        return "Searching"
    elif msg in fun_list:
        exec(fun_list[msg])
        return("Executing")
    else:
        sentence = tokenize(msg)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])
        

        return "I dont understand..."


def google():
    wb.open("https://www.google.es/")
    return ("Opening Google")

def search_google(msg):
    x= msg.partition(' ')[2]
    print(x)
    wb.open(f"https://www.google.com/search?q={x}")

def youtube():
    wb.open("https://www.youtube.com/")


def monkeytype():
    wb.open("https://www.monkeytype.com/")

def cleansing():
    data ="""
import pandas as pd
import numpy as np
import regex as re
    """
    os.system('code test.ipynb')
    subprocess.run("pbcopy", text=True, input=data)
def visuals():
    data ="""
import pandas as pd
import numpy as np
import regex as re
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Matplotlib inline to visualize Matplotlib graphs
%matplotlib inline
%config Inlinebackend.figure_format= 'retina'

sns.set_context("poster")
sns.set(rc={"figure.figsize": (12.,6.)})
sns.set_style("whitegrid")      
    """
    os.system('code test.ipynb')
    subprocess.run("pbcopy", text=True, input=data)

# if its time
def sms():
    pass