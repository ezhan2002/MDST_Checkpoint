#!/usr/bin/env python
# coding: utf-8

# # Checkpoint 0

# These exercises are a mix of Python and Pandas practice. Most should be no more than a few lines of code!

# In[ ]:


# here is a Python list:

a = [1, 2, 3, 4, 5, 6]


# In[ ]:


# get a list containing the last 3 elements of a
# Yes, you can just type out [4, 5, 6] but we really want to see you demonstrate you know how to use list slicing in Python
last_3_elements = a[-3:]
print(last_3_elements)


# In[4]:


print(list(range(1,21)))


# In[5]:


# now get a list with only the even numbers between 1 and 100
# you may or may not make use of the list you made in the last cell
list(number for number in range(2, 101, 2))


# In[6]:


# write a function that takes two numbers as arguments
# and returns the first number divided by the second
def func(a, b):
    return a / b

#examples
func(12,6)


# In[7]:


# write a function that takes a string as input
# and return that string in all caps
def func1(s):
    return s.upper()

##example
func1("happy")


# In[8]:


# fizzbuzz
# you will need to use both iteration and control flow
# go through all numbers from 1 to 30 in order
# if the number is a multiple of 3, print fizz
# if the number is a multiple of 5, print buzz
# if the number is a multiple of 3 and 5, print fizzbuzz and NOTHING ELSE
# if the number is neither a multiple of 3 nor a multiple of 5, print the number

def func3(a):
    if a % 3 == 0 and a % 5 == 0:
        print("fizzbuzz")
    elif a % 3 == 0:
        print("fizz")
    elif a % 5 == 0:
        print("buzz")
    else:
        print(a)

for i in range(1, 31):
    func3(i)


# In[28]:


# create a dictionary that reflects the following menu pricing (taken from Ahmo's)
# Gyro: $9
# Burger: $9
# Greek Salad: $8
# Philly Steak: $10

out= {
    "Gyro" : 9,
    "Burger" : 9,
    "Greek Salad" : 8,
    "Philly Steak" : 10
}

print(out)


# In[1]:


# load in the "starbucks.csv" dataset
# refer to how we read the cereal.csv dataset in the tutorial
import pandas as pd
data = pd.read_csv("/home/ezhan/MDST_checkpoints/starbucks.csv")
print(data.head())


# In[24]:


# output the calories, sugars, and protein columns only of every 40th row.
output = data.iloc[::40][["calories", "sugars", "protein"]]
print(output)
 


# In[11]:


# select all rows with more than and including 400 calories
high = data[data["calories"] >= 400]
print(high)


# In[26]:


# select all rows whose vitamin c content is higher than the iron content
higher_vc = data[data["vitamin c"] > data["iron"]]
print(higher_vc)


# In[27]:


print(len(higher_vc))


# In[13]:


# create a new column containing the caffeine per calories of each drink
data["caffeine_per_calories"] = data["caffeine"] / data["calories"]


# In[14]:


# what is the average calorie across all items?
avg = data["calories"].mean()
print(avg)


# In[19]:


# how many different categories of beverages are there?
diff = data["beverage_category"].unique()
print(len(diff))


# In[ ]:


# what is the average # calories for each beverage category?
avg_count = data.groupby("beverage_category")["calories"].mean()
print(avg_count)


# In[30]:


# plot the distribution of the number of calories in drinks with a histogram
import matplotlib.pyplot as plt
data["calories"].hist(bins=30, edgecolor = "blue")

plt.title("distribution of # of calories in drinks")
plt.xlabel("calorie")
plt.ylabel("drink")
plt.show()


# In[32]:


# plot calories against total fat with a scatterplot
plt.scatter(data["calories"], data["total fat"], alpha = 0.4)

plt.xlabel("calories")
plt.ylabel("total_fat")
plt.grid(True)
plt.show()


# In[ ]:




