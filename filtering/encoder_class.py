from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 
from sentence_transformers import CrossEncoder

class filter(object):

    def scorer(question:str, answers:list):
        
        model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
        tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')

        features = tokenizer([question] * len(answers), answers,  padding=True, truncation=True, return_tensors="pt")

        model.eval()
        with torch.no_grad():
            scores = model(**features).logits
        return scores

    def sentence_scorer(question:str, answers:list):
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', max_length=512)
        tuples = []
        for a in answers:
            tup = (question, a)
            tuples.append(tup)
        scores = model.predict(tuples)
        return scores