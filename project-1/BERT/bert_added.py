# Downloaded all the necessary libararies
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import torch

# Loading the training and testing data from CSV files
train_df = pd.read_csv('C:\\Users\\elmed\\Downloads\\train.csv')
test_df = pd.read_csv('C:\\Users\\elmed\\Downloads\\test.csv')

# Filling missing values in 'keyword' and 'location' columns with empty strings
train_df['keyword'].fillna('', inplace=True)
train_df['location'].fillna('', inplace=True)

# Combining 'keyword', 'location', and 'text' columns into a single text column for processing
train_df['combined_text'] = train_df['keyword'] + " " + train_df['location'] + " " + train_df['text']
test_df['combined_text'] = test_df['keyword'] + " " + test_df['location'] + " " + test_df['text']

# Preprocessing the combined text by converting to lowercase and removing special characters
train_df['combined_text'] = train_df['combined_text'].str.lower().str.replace('[^a-z0-9\s]', '', regex=True)
test_df['combined_text'] = test_df['combined_text'].str.lower().str.replace('[^a-z0-9\s]', '', regex=True)

# Splitting the training data into a training set and a validation set for model evaluation
train_text, val_text, train_labels, val_labels = train_test_split(
    train_df['combined_text'], train_df['target'], test_size=0.1
)

# Initializing the tokenizer from the DistilBERT model, which is a lighter version of BERT
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenizing the training and validation text
# This will convert text to a format that the model can understand
train_encodings = tokenizer(train_text.tolist(), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_text.tolist(), truncation=True, padding=True, max_length=128)

# Defining a custom dataset class to handle the PyTorch data loading
class DisasterDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# Creating dataset objects for training and validation
train_dataset = DisasterDataset(train_encodings, train_labels.tolist())
val_dataset = DisasterDataset(val_encodings, val_labels.tolist())

# Initializing the DistilBERT model for sequence classification with two labels
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Defining a function to compute evaluation metrics (F1 score and accuracy)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}

# Setting up training arguments, including directory for results, number of epochs, batch size, etc.
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Initializing the Trainer with the model, training arguments, datasets, and evaluation metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Training the model on the dataset
trainer.train()

# Also downloaded transformers 
# pip install torch