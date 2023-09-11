#!/usr/bin/env python
# coding: utf-8

# # Checkpoint 1

# Reminder: 
# 
# - You are being evaluated for compeletion and effort in this checkpoint. 
# - Avoid manual labor / hard coding as much as possible, everything we've taught you so far are meant to simplify and automate your process.

# We will be working with the same `states_edu.csv` that you should already be familiar with from the tutorial.
# 
# We investigated Grade 8 reading score in the tutorial. For this checkpoint, you are asked to investigate another test. Here's an overview:
# 
# * Choose a specific response variable to focus on
# >Grade 4 Math, Grade 4 Reading, Grade 8 Math
# * Pick or create features to use
# >Will all the features be useful in predicting test score? Are some more important than others? Should you standardize, bin, or scale the data?
# * Explore the data as it relates to that test
# >Create at least 2 visualizations (graphs), each with a caption describing the graph and what it tells us about the data
# * Create training and testing data
# >Do you want to train on all the data? Only data from the last 10 years? Only Michigan data?
# * Train a ML model to predict outcome 
# >Define what you want to predict, and pick a model in sklearn to use (see sklearn <a href="https://scikit-learn.org/stable/modules/linear_model.html">regressors</a>.
# * Summarize your findings
# >Write a 1 paragraph summary of what you did and make a recommendation about if and how student performance can be predicted
# 
# Include comments throughout your code! Every cleanup and preprocessing task should be documented.
# 
# Of course, if you're finding this assignment interesting (and we really hope you do!), you are welcome to do more than the requirements! For example, you may want to see if expenditure affects 4th graders more than 8th graders. Maybe you want to look into the extended version of this dataset and see how factors like sex and race are involved. You can include all your work in this notebook when you turn it in -- just always make sure you explain what you did and interpret your results. Good luck!

# <h2> Data Cleanup </h2>
# 
# Import `numpy`, `pandas`, and `matplotlib`.
# 
# (Feel free to import other libraries!)

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None


# Load in the "states_edu.csv" dataset and take a look at the head of the data

# In[3]:


df= pd.read_csv("/home/ezhan/MDST_checkpoints/states_edu.csv")
df.head()


# You should always familiarize yourself with what each column in the dataframe represents. Read about the states_edu dataset here: https://www.kaggle.com/noriuk/us-education-datasets-unification-project

# Use this space to rename columns, deal with missing data, etc. _(optional)_

# In[4]:


df.isna().sum()


# In[5]:


df.dropna()


# <h2>Exploratory Data Analysis (EDA) </h2>

# Chosen Outcome Variable for Test: *Avg_Math_8_Score*

# How many years of data are logged in our dataset? 

# In[6]:


df["YEAR"].nunique()


# Let's compare Michigan to Ohio. Which state has the higher average outcome score across all years?

# In[7]:


michigan_df = df[df['STATE'] == 'MICHIGAN']


michigan_df['AVG_MATH_8_SCORE'].mean()


# In[8]:


ohio_df = df[df['STATE'] == 'OHIO']
ohio_df['AVG_MATH_8_SCORE'].mean()


# Find the average for your outcome score across all states in 2019

# In[9]:


df_2019 = df[df['YEAR'] == 2019] 
df_2019['AVG_MATH_8_SCORE'].mean()


# Find the maximum outcome score for every state. 
# 
# Refer to the `Grouping and Aggregating` section in Tutorial 0 if you are stuck.

# In[10]:


df.groupby('STATE')['AVG_MATH_8_SCORE'].max()


# <h2> Feature Engineering </h2>
# 
# After exploring the data, you can choose to modify features that you would use to predict the performance of the students on your chosen response variable. 
# 
# You can also create your own features. For example, perhaps you figured that maybe a state's expenditure per student may affect their overall academic performance so you create a expenditure_per_student feature.
# 
# Use this space to modify or create features.

# In[13]:


df['TOTAL_EXPENDITURE_PER_8'] = df['TOTAL_EXPENDITURE'] / df['GRADES_8_G']
df.head


# In[66]:


df["TOTAL_FED_REVENUE_PER_8"] = df["FEDERAL_REVENUE"] / df["GRADES_8_G"]


# Feature engineering justification: **<BRIEFLY DESCRIBE WHY YOU MADE THE CHANGES THAT YOU DID\>**
# I have calculated the total expenditure that the schools spend on each 8th grade student to deremine if there are any correlations between the amount of money spend per student and his/her respective math and English scores. 
# I have created the TOTAL_FED_REVENUE_PER_8 as a new column that calculates the total Federal Revenue per 8th graders to be used later in the ML analysis. 

# <h2>Visualization</h2>
# 
# Investigate the relationship between your chosen response variable and at least two predictors using visualizations. Write down your observations.
# 
# **Visualization 1**

# In[18]:


plt.figure(figsize=(8, 6))
plt.scatter(df["TOTAL_EXPENDITURE_PER_8"], df["AVG_MATH_8_SCORE"], alpha=0.7)
plt.xlabel('Expenditure per 8th Grader')
plt.ylabel('Average 8th Grade Math Score')


# Plot of the relationship between Average Combined Score and Total Expenditure for 8th graders

# **Visualization 2**

# In[19]:


plt.figure(figsize=(8, 6))
plt.scatter(df["TOTAL_EXPENDITURE_PER_8"], df["AVG_READING_8_SCORE"], alpha=0.7)
plt.xlabel('Expenditure per 8th Grader')
plt.ylabel('Average 8th Grade Reading Score')


# Based on my plot, it seems that there are some correlations with expenditures per 8th grader and their English and Math Levels as we observe, especially in the case of the Math Score, that the higher the expenditure per student, the higher the Math score. 

# <h2> Data Creation </h2>
# 
# _Use this space to create train/test data_

# In[78]:


from sklearn.model_selection import train_test_split


# In[79]:


X = df[['GRADES_8_G','TOTAL_EXPENDITURE_PER_8','TOTAL_FED_REVENUE_PER_8']].dropna()
y = df.loc[X.index]['AVG_MATH_8_SCORE']
y.fillna(y.median(), inplace=True)


# In[80]:


X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size= 0.3, random_state=42)


# <h2> Prediction </h2>

# ML Models [Resource](https://medium.com/@vijaya.beeravalli/comparison-of-machine-learning-classification-models-for-credit-card-default-data-c3cf805c9a5a)

# In[81]:


from sklearn.linear_model import LinearRegression


# In[82]:


model = LinearRegression()


# In[83]:


model.fit(X_train, y_train)


# In[84]:


y_pred = model.predict(X_test)


# ## Evaluation

# Choose some metrics to evaluate the performance of your model, some of them are mentioned in the tutorial.

# In[85]:


model.score(X_test, y_test)


# In[86]:


np.mean(model.predict(X_test)-y_test)


# We have copied over the graphs that visualize the model's performance on the training and testing set. 
# 
# Change `col_name` and modify the call to `plt.ylabel()` to isolate how a single predictor affects the model.

# In[87]:


col_name = 'TOTAL_EXPENDITURE_PER_8'

f = plt.figure(figsize=(12,6))
plt.scatter(X_train[col_name], y_train, color = "red")
plt.scatter(X_train[col_name], model.predict(X_train), color = "green")

plt.legend(['True Training','Predicted Training'])
plt.xlabel(col_name)
plt.ylabel('AVG_MATH_8_SCORE')
plt.title("Model Behavior On Training Set")


# In[88]:


col_name = 'TOTAL_FED_REVENUE_PER_8'

f = plt.figure(figsize=(12,6))
plt.scatter(X_test[col_name], y_test, color = "blue")
plt.scatter(X_test[col_name], model.predict(X_test), color = "black")

plt.legend(['True testing','Predicted testing'])
plt.xlabel(col_name)
plt.ylabel('AVG_READING_8_SCORE')
plt.title("Model Behavior on Testing Set")


# <h2> Summary </h2>

# I am trying to predict the average 8th graders math score by using the total expenditures per student and the federal revenue per student. From the graph, it seems that there are some positive correlation between total expenditure per student and average 8th graders math score. However, the ML model fails to produce the linear model. Whereas the other two variables, the Total Federal Revenue per 8th graders and the Avergae math scores yields little linear relationships. The matrices used, both the R^2 value and the mean error, yields a far greater error. Therefore, I have concluded that the expenditures per students, whethere it is total expenditures or federal revenue, are not good predictors for prediciting student's average math scores. Other measures, such as a combination of local and federal revenues and expenditures, such as the ratio of Federal Revenue in Total Expenditures, could be used in the future for better predictors. 
