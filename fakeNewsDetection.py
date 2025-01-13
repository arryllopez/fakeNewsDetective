#importing the necessary libraries
import pandas as pd
import numpy as np

#loading the dataset
fake_dataset = pd.read_csv('archive/datasets/Fake.csv')
true_dataset = pd.read_csv('archive/datasets/True.csv')

#testing to see if loading is succesful 
print (fake_dataset.head())