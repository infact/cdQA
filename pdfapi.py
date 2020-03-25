from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
from ast import literal_eval
import pandas as pd
from cdqa.utils.converters import pdf_converter
from cdqa.utils.filters import filter_paragraphs
from cdqa.utils.download import download_model
from cdqa.pipeline import QAPipeline
from flask import make_response
import traceback

app = Flask(__name__)
CORS(app)
def application(environ, start_response):
    if environ['REQUEST_METHOD'] == 'OPTIONS':
        start_response(
    '200 OK',
    [
        ('Content-Type', 'application/json'),
        ('Access-Control-Allow-Origin', '*'),
        ('Access-Control-Allow-Headers', 'Authorization, Content-Type'),
        ('Access-Control-Allow-Methods', 'GET','POST'),
    ]
    )
    return ''
os.environ["dataset_path"]="./data/pdf/"
os.environ["reader_path"]="./models/bert_qa.joblib"

dataset_path = os.environ["dataset_path"]
reader_path = os.environ["reader_path"]
print('-----------------')
print(dataset_path)
print(reader_path)
print('-----------------')
df = pdf_converter(directory_path = dataset_path)
print('-----------------')
print(df)
print('-----------------')
df.head()
#df = filter_paragraphs(df)

cdqa_pipeline = QAPipeline(reader=reader_path, max_df=1.0)
cdqa_pipeline.fit_retriever(df=df)


@app.route("/api", methods=["GET"])
def api():
    query = request.args.get("query")
    print(query)
    prediction = cdqa_pipeline.predict(query=query)
    return jsonify(
        query=query, answer=prediction[0], title=prediction[1], paragraph=prediction[2]
    )
    
if __name__ == '__main__':
    app.run(debug=True)
