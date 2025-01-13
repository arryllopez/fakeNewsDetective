#importing the necessary libraries
import pandas as pd
import numpy as np

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