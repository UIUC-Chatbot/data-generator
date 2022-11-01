from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 

class filter(object):

    def scorer(question, answers):
        
        model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
        tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')

        features = tokenizer([question] * len(answers), answers,  padding=True, truncation=True, return_tensors="pt")

        model.eval()
        with torch.no_grad():
            scores = model(**features).logits
        return scores