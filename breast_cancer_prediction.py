import pandas as pd   #importing libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer 
cancer = load_breast_cancer()  #loading dataset
cancer
df = pd.DataFrame(cancer.data, columns = cancer.feature_names)  #creating a dataframe
df
df['label'] = cancer.target
df
df.info()
df.describe().T
sns.countplot(x = df.label)    #plotting number of data in each category
df.corr()
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True)   #plotting confusion matrix in a heatmap
X = df.drop('label',axis=1)
y = df['label']
X
y
from sklearn.model_selection import train_test_split     #importing library
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)  #splitting data into training and testing data
df.shape
X_train.shape
X_test.shape
from sklearn.linear_model import LogisticRegression  #importing logistic regression library
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
print(classification_report(y_test,y_pred))
accuracy_score(y_test, y_pred) * 100
cm = confusion_matrix(y_test,y_pred)
cm
sns.heatmap(cm,annot=True)

#adding new data for testing
new_data = [[15.3, 18.2, 102.7, 700.2, 0.09, 0.12, 0.1, 0.06, 0.18, 0.07, 
             0.3, 1.2, 2.1, 20.0, 0.005, 0.02, 0.03, 0.01, 0.02, 0.005, 
             17.3, 25.5, 115.6, 900.2, 0.12, 0.19, 0.14, 0.07, 0.2, 0.09]]

prediction = model.predict(new_data)

if prediction[0] == 0:
    print("The tumor is cancerous.")
else:
    print("The tumor is not cancerous.")
print("\n\n\n")
malignant_sample = [[20.57, 17.77, 132.9, 1326.0, 0.08474, 0.07864, 0.0869, 0.07017, 0.1812, 0.05667, 
                     0.5435, 0.7339, 3.398, 74.08, 0.005225, 0.01308, 0.0186, 0.0134, 0.01389, 0.003532, 
                     24.99, 23.41, 158.8, 1956.0, 0.1238, 0.1866, 0.2416, 0.186, 0.275, 0.08902]]


prediction = model.predict(malignant_sample)

if prediction[0] == 0:
    print("The tumor is cancerous.")
else:
    print("The tumor is not cancerous.")
print("\n\n\n")