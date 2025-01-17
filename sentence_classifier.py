import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import pickle

def preprocess_function(examples):
    model_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Perform predictions
def predict(model, dataset):
    model.eval()
    predictions = []
    for batch in dataset:
        # Filter only the required keys for the model
        inputs = {key: value.unsqueeze(0).to(model.device) if isinstance(value, torch.Tensor) else value 
                  for key, value in batch.items() if key in ['input_ids', 'attention_mask']}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).item()
        predictions.append(pred)
    
    return predictions

def get_results():
    label_mapping = {
        'general-url': 0,
        'third-party-dataset': 1,
        'author-provided-dataset': 2,
        'third-party-software': 3,
        'author-provided-software': 4,
        'project': 5
    }
    model_name = "allenai/scibert_scivocab_uncased"
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_mapping))

    # Load the saved model state dictionary
    model_file_path = 'best_model-v1.pkl'  # Replace with your model file path
    with open(model_file_path, 'rb') as f:
        state_dict = pickle.load(f)
    model.load_state_dict(state_dict)
    
    test_csv_path = 'OADS_Test_Sentence.csv'  # Replace with your test CSV file path
    test_df = pd.read_csv(test_csv_path)
    test_dataset = Dataset.from_pandas(test_df)

    # Map the tokenization function to the dataset
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    # Set format to PyTorch
    test_dataset.set_format('torch')

    # Predict the labels for the test data
    predictions = predict(model, test_dataset)




    return predictions
