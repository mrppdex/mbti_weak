from flask import Flask, escape, request, render_template
import sys, os

sys.path.append(os.path.realpath(os.path.curdir))

from pipe2 import mtbi_inference

app = Flask(__name__)

@app.route("/")
def home():
  return render_template("index.html")

@app.route("/mbti", methods=['POST'])
def mbti_inference():
  text = escape(request.form['text'])
  ptypes = ["Extrovert", "Introvert"]
  inference_object = mtbi_inference(text, types=ptypes)
  pred = inference_object.predict()
  del inference_object
  return f"You are {max(pred, 1-pred)*100:.2f}% {ptypes[int(pred>=0.5)]}..."
  


if __name__ == "__main__":
    app.run(debug=True)
