from flask import Flask, request, render_template, redirect
import torch
import pickle
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("fine_tuned_sbert_model")
corpus_embeddings = torch.load("corpus_embeddings.pt")
with open("sentences.pkl", "rb") as f:
    corpus = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods = ['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        # print(request.form)
        sentence = request.form['Sentence']
        top5 = searchTop5(sentence)
        return render_template('index.html', top5 = top5)
    return render_template('index.html')

def searchTop5(sentence):
    query_embedding = model.encode(sentence, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(scores, k=5)
    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        results.append({
            "sentence": corpus[idx],
            "score": round(score.item(), 4)
        })

    return results

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)