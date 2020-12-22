from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from rank_bm25 import BM25Okapi

annotated_df = pd.read_csv('aggregated-hw3-ratings.train.csv', header=None)
annotated_df.columns = ['message_id', 'response_id', 'score']
annotated_df.drop(columns='message_id', inplace=True)

chat_response = pd.read_csv('chatbot-replies.tsv.gz', sep='\n\t|\t', engine='python',compression='gzip', encoding='utf8')

df = chat_response.merge(annotated_df, how='inner', on=['response_id'])
response_df = [x for x in df.response]

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/get")
def get_bot_response():
    what_the_user_said = request.args.get('msg')
    query_tokenized = what_the_user_said.split(" ")

    tokenized_response = [doc.split(" ") for doc in response_df]
    bm25 = BM25Okapi(tokenized_response)
    doc_scores = bm25.get_scores(query_tokenized)
    results = bm25.get_top_n(query_tokenized, response_df, n=1)
    result = results[0]
    return str(result)

if __name__ == "__main__":

    
    # IMPLEMENTATION HINT: you probably want to load and cache your conversation
    # database (provided by us) here before the chatbot runs
       
    app.run()
