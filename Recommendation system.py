import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import sklearn
from sklearn.decomposition import TruncatedSVD

#loading dataset
df = pd.read_csv('ratings_Beauty.csv')
df = df.dropna()
df1 = df.head(10000)

utility_matrix = df1.pivot_table(values='Rating', index='UserId', columns='ProductId', fill_value=0)

#Transposing the utility matrix
X = utility_matrix.T
X1 = X
SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)

correlation_matrix = np.corrcoef(decomposed_matrix)

#Assume customer buys productID 6117036094 
i = "6117036094"
product_names = list(X.index)
product_ID = product_names.index(i)

correlation_product_ID = correlation_matrix[product_ID]

Recommend = list(X.index[correlation_product_ID > 0.90])
Recommend.remove(i)

# top 10 products to be displayed by the recommendation system to the above customer based on the purchase history of other customers in the website
Recommend[0:9]
