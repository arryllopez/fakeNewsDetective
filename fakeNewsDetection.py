#importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from IPython.display import FileLink
import os 
import shutil

#loading the dataset
fake_dataset = pd.read_csv('archive/datasets/Fake.csv')
true_dataset = pd.read_csv('archive/datasets/True.csv')

#testing to see if loading is succesful 
print (fake_dataset.head())
print (true_dataset.head())
print(fake_dataset.describe(include='object'))
print(true_dataset.describe(include='object'))

#adding a new column assigning a constant value to each dataset 
fake_dataset["label"] = "fake"
true_dataset["label"] = "true"

#testin if label is applied
print(fake_dataset.head())
print(true_dataset.head())


#shuffling the rows of each dataset so that the combined data set is randomized between false and true 
combined_data = pd.concat ([fake_dataset, true_dataset])
combined_data = combined_data.sample(n=len(combined_data), random_state=None).reset_index(drop=True) 
combined_data

#printing to see if it is shuffled
print(combined_data)


#checking to see if there are any empty values in the columns
print(combined_data.isnull().sum()) 

#number of fake pieces of data and number of pieces of true data
print(combined_data.label.value_counts()) 

#matplot plotting the count of fake vs real news
label_counts=combined_data.label.value_counts()
plt.figure(figsize=(6,4))
#describing the style of the plot an the colors 
label_counts.plot(kind='bar', color=['red', 'green'])
plt.title('Count of Fake vs Real News', fontsize=14)
plt.xlabel('Label', fontsize=12)
plt.ylabel('Count', fontsize=12)
#showing the graph 
plt.show() 

#attaching a lable map of true=1 false= 0
combined_data["input"] = combined_data["title"] + "" +combined_data["text"]
combined_data=combined_data[["input","label"]]
combined_data = combined_data.copy()
combined_data['label'] = combined_data['label'].map({'true' : 1, 'fake':0})
print(combined_data.sample(7))  

#----------------------------------------------------------

#tokenization and processing the data
ds = Dataset.from_pandas(combined_data)
print(ds) 

#importing the model (pretrained microsoft deBERTa)
model = 'microsoft/deberta-v3-small'
tokenizer = AutoTokenizer.from_pretrained(model) 

def tok_func(x): 
    return tokenizer(x["input"], padding="max_length", truncation=True, max_length=512)

tok_ds = ds.map(tok_func, batched=True)

print(tok_ds)

row = tok_ds[0]
row['input'], row['input_ids']

tok_ds = tok_ds.rename_columns({'label':'labels'})

tokenizer.tokenize(row['input'])

dds = tok_ds.train_test_split(0.25, seed=42)

def accuracy_metric(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}

bs = 16
epochs = 4
lr = 2e-5

args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
    eval_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
    num_train_epochs=epochs, weight_decay=0.01, report_to='none')

fake_news_model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=2)

trainer = Trainer(fake_news_model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                  tokenizer=tokenizer, compute_metrics=accuracy_metric)


trainer.train();        

fake_news_model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

import os
print(os.listdir("./saved_model"))

shutil.make_archive("saved_model", 'zip', "./saved_model")

FileLink("./saved_model.zip")