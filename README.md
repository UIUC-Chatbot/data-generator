
# data-generator

### HF
Models used from HuggingFace.co
* `huggingface.ipynb`
	* `end2end.json` - ThomasSimonini/t5-end2end-question-generation
	* `fine_tuned_data.json` - mrm8488/t5-base-finetuned-question-generation-ap
	* Dataset Quality - Not Good

### bart

Inspired from this [paper's](https://arxiv.org/abs/2102.12128) second best model using bart question generator and bert-mrc

* `bart.ipynb`
* Two different bert-mrc
	* `bespin.json` 
	* `ainze.json`
	* Dataset Quality - Not Good

### filtering
* `cross_encoder.ipynb`
	* Uses this [model](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2?text=I+like+you.+I+love+you) on HuggingFace.
* `scoring.ipynb`
	* Uses bleu scores
* `Dialogue_RPT_Scoring.ipynb`
	* Uses Dialogue RPT to rate the asnwers based on the Context of the GPT answer file
