#importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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