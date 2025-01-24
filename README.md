# ACD: Adversarial Counterfactual Distillation for Rating Debiasing in Recommendation
This is our PyTorch implementation for the paper.


## Requirements

```
python == 3.8.3
transformers == 3.1.0
dgl == 0.7.2
pytorch == 1.10.2
```

## Running the code

1. Run word2vector.py for word embedding. Glove pretraining weight is required. 
2. Make sure can run load_sentiment_data in load_data.py 
3. Run BERT/bert_whitening.py for obtaining the feature vector for each review.
4. If previous steps successfully run, then you can run ACD.py. 
