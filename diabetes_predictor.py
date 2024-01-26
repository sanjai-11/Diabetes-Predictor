# Importing required libraries for analysis and visulization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing file from a system
from google.colab import files
uploaded = files.upload()

# Reading the file
df = pd.read_csv("diabetes.csv")
df.head(5)

# Check the rows and Columns
df.shape

# Display the column names in the Dataframe
df.columns

# Display information about the Dataframe
df.info()

"""**Checking the Missing Values**
--- 
"""

# Check Null values in a tabular form
df.isnull().sum()

# Generate summary statitics for the numerical columns in the Dataframe
df.describe().transpose()

# Create a heatmap to visualize missing values (NULL) in the Dataframe
sns.heatmap(df.isnull(),cmap="viridis")

# Calculate the correlation matrix for the Dataframe columns
correlation = df.corr()
print(correlation)

"""# **CORRELATION MATRIX**

1. Glucose and Age have moderate positive correlations with Outcome, indicating that higher glucose levels and older age are associated with a higher likelihood of diabetes.

2. BMI and Insulin also show some positive correlation with Outcome, though weaker that Glucose and Age.

3. Most other variables have relatively low correlations with Outcome.

As shown in the plot above, there are no null values present.
"""

# Check the heatmap correlation of the features
sns.heatmap(df.corr(), annot = True, cmap = "YlGnBu")

"""# **DATA VISUALIZATION**
---

**Distribution of data using appropriate plots.**
"""

# Create histogram for each column in the Dataframe with specified parameters
df.hist(bins = 20,figsize=(10,10),color='cyan',edgecolor='k',alpha=1,lw=1)
plt.show()

# Calculate skewness for the dataframe
def skewness(data):
  '''This function calcultes the skewness value
  of each column
  parameter: data=dataframe
  return: Series with skewness values for each column'''
  return data.skew()

skewness(df)

"""By calling defined function, i calculate skewness of each column of dataframe.

If skewness is between **-0.5 to 0.5 then data are fairly symmetrical**

If skewness is between **-1 to -0.5 & 0.5 to 1 then data are moderatelu skewed**.

If skewness is between **lt -1 or gt 1 then data are highly skewed.**

**To find the outliers present in a given dataset using box plot method**
"""

c_palette = {0: 'black', 1: 'orange'}

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Plot the first boxplot on the left subplot
sns.boxplot(x="Outcome", y="Pregnancies", data=df, palette=c_palette, ax=axes[0])
axes[0].set_title("Diabetes by Pregnancies")

# Plot the second boxplot on the right subplot
sns.boxplot(x="Outcome", y="Glucose", data=df, palette=c_palette, ax=axes[1])
axes[1].set_title("Glucose Distribution by Diabetes Status")

# Display the plots
plt.show()

"""This box plot illustrates the distribution of the number of pregnancies for diabetic(1) and non-diabetic(0) patients.

The non-diabetic group(0) exhibits outliers

**Diabetic patients tend to have higher glucose levels compared to non-diabetic patient,** there are some exceptioal case with elevated glucose levels.
"""

c_palette = {0:"black", 1:"orange"}

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Plot the first boxplot for "BloodPressure" on the left subplot
sns.boxplot(x="Outcome", y="BloodPressure", data=df, palette=c_palette, ax=axes[0])
axes[0].set_title("BloodPressure Distribution by Diabetes Status")

# Plot the second boxplot for "Age" on the right subplot
sns.boxplot(x="Outcome", y="Age", data=df, palette=c_palette, ax=axes[1])
axes[1].set_title("Age Distribution by Diabetes Status")

# Display the plots
plt.show()

"""**Diabetic individuals (orange) tend to have a slightly higher blood pressure** compared to non-diabetic individuals (black).

Diabetic individuals (orange) are generally younger, while non-diabetic individuals (black) span a wider age range.
"""

c_palette = {0: 'black', 1: 'orange'}

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(10,4))

# Plot the boxplot for "BMI" on the left subplot
sns.boxplot(x="Outcome", y="BMI", data=df, palette=c_palette, ax=axes[0])
axes[0].set_title(" BMI Distributions by Diabetes Status")

# Plot the boxplot for "Insulin" on the right subplot
sns.boxplot(x="Outcome", y="Insulin", data=df, palette=c_palette, ax=axes[1])
axes[1].set_title("Insulin Distribution by Diabetes Status")

# Display the plots
plt.show()

"""There's a noticeable difference in BMI distribution between the two groups, with **diabetic individuals having higher BMI on average.**

**Diabetic individuals show more variation in insulin levels**(orange) that non-diabetics(black).

**Glucose:** The Glucose column does not show significant outliers, and it has a relatively symmetrical distribution.

**Outcome:** The "Outcome" column is a binary variable and is not visualized in a box plot.

And rest columns shows a skewed distributions with some outliers

**Distribution Of Outcome**

---
"""

# Calculate the number of diabetic individuals (outcoe-1) and non-diabetic individuals (outcome-0)
diabetic = len(df[df["Outcome"]==1])
non_diabetic = len(df[df["Outcome"]==0])

# Create a counts of diabetic and non-diabetic individuals
count = (diabetic, non_diabetic)

# Define labels for the pie chart, representing the categories
labels = ("Diabetic", "Non Diabetic")

plt.pie(count, labels=labels, autopct="%1.1f%%", colors={"orange", "black"})

# Set a title for the pie chart
plt.title("Diabetic vs. Non-diabetic Ratio")

plt.show()

"""**Importing Libraries for Machine Learning Model**

---
"""

# Import necessary libraries and suppress warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

# Splitting Data into training and testing sets
x = df.drop('Outcome', axis=1)
y = df['Outcome']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

# Creating and Training a Logistic Regression Model
model = LogisticRegression()
model.fit(x_train,y_train)

# Making Predictions and Printing the results
prediction = model.predict(x_test)
print(prediction)

# Calculating and Printing Model Accuracy
accuracy = accuracy_score(prediction,y_test)
print(accuracy)