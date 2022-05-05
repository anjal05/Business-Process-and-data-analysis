#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 09:51:40 2022

@author: anjalisaini
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import squarify
from apyori  import apriori

groceries= pd.read_csv("groceries.csv", header=None)
groceries.head()
print(groceries.shape)

# looking at the frequency of most popular items 

plt.rcParams['figure.figsize'] = (18, 7)
color = plt.cm.copper(np.linspace(0, 1, 40))
groceries[0].value_counts().head(40).plot.bar(color = color)
plt.title('frequency of most popular items', fontsize = 20)
plt.xticks(rotation = 90 )
plt.grid()
plt.show()

y = groceries[0].value_counts().head(50).to_frame()
y.index
# plotting a tree map

plt.rcParams['figure.figsize'] = (20, 20)
color = plt.cm.cool(np.linspace(0, 1, 50))
squarify.plot(sizes = y.values, label = y.index, alpha=.8, color = color)
plt.title('Tree Map for Popular Items')
plt.axis('off')
plt.show()

unique_items        = (groceries[0].unique())#checking unique values
print(unique_items)


records = []
for i in range(0, 7501):
    records.append([str(groceries.values[i,j]) for j in range(0, 20)])



'''The Apriori algorithm is one of the most common techniques in Market Basket Analysis. 
It is used to analyze the frequent itemsets in a transactional database, which then is used 
to generate association rules between the products. The association rules exploration is 
based on the idea that the purchasing behavior of customers follows a pattern that can be 
used to sell more products to the customer in the future.

Main Concepts of Association Rules / Apriori Algorithm:
Support is a measure of frequency of the itemset that appears in the dataset. This is an 
indication of how popular an itemset is in a dataset.

Confidence is a measure of the reliability of the rule. Is an indication of how often the 
rule has been found to be true. It can be said confidence says how likely item Y is purchased 
when item X is purchased.

Lift shows how likely item Y is purchased when item X is purchased, while controlling for how 
popular item Y is.'''



association_rule_1 = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_result_1 = list(association_rule_1)
print(association_result_1[0])
'''We can see that baking powder and whiped/sour cream are purchased together. This makes sense as 
these both are the ingredients for baking cake. 
The support value for the first rule is 0.0050. This number is calculated by dividing the 
number of transactions containing whiped/sour cream divided by total number of transactions. 
The confidence level for the rule is 0.2835 which shows that out of all the transactions that 
contain whiped/sour cream, 28.35% of the transactions also contain baking powder. 
Finally, the lift of 3.88 tells us that baking powder is 3.88 times more likely to be bought by the 
customers who buy whiped/sour cream compared to the default likelihood of the sale of baking powder'''


association_rule_2 = apriori(records, min_support=0.0065, min_confidence=0.3, min_lift=4, min_length=3)
association_result_2 = list(association_rule_2)
print(association_result_2[0])
'''We can see that tropical fruit and root vegetables are purchased together. Also whole milk 
and other vegetables are bought together.
The support value for the first rule is 0.0074. This number is calculated by dividing the 
number of transactions containing tropical fruit divided by total number of transactions. 
The confidence level for the rule is 0.3373 which shows that out of all the transactions that 
contain tropical fruits 33.73% of the transactions also contain root vegetables. 
Finally, the lift of 4.58 tells us that root vegetable is 4.58 times more likely to be bought by the 
customers who buy tropical fruits compared to the default likelihood of the sale of root vegetable'''



association_rule_3 = apriori(records, min_support=0.0085, min_confidence=0.4, min_lift=2, min_length=2)
association_result_3 = list(association_rule_3)
print(association_result_3[0])
'''We can see that baking powder and whole milk are purchased together. This makes sense as 
these both are the ingredients for baking cake. 
The support value for the first rule is 0.0093. This number is calculated by dividing the 
number of transactions containing whole milk divided by total number of transactions. 
The confidence level for the rule is 0.5223 which shows that out of all the transactions that 
contain whiole milk 52.23% of the transactions also contain baking powder. 
Finally, the lift of 2.05 tells us that baking powder is 2.05 times more likely to be bought by the 
customers who buy whole milk compared to the default likelihood of the sale of baking powder'''

#At this point, you can see how great the possibilities are to use the popularity of one product to increase the sales of another








