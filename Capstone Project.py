#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, classification_report, roc_auc_score, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_curve

# Plotting packages.
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


import random
random.seed(11381823)


# In[16]:


df = pd.read_csv('musicData.csv')
df = df.dropna()
df = df.drop(['obtained_date'], axis = 1)
df.drop(['artist_name','track_name'], axis = 1)

df = pd.get_dummies(df, columns = ['key'], dtype=int)


# In[17]:


df = df[
    (df['popularity'] >= 0) & (df['popularity'] < 100) &
    (df['acousticness'] < 100) &
    (df['danceability'] >= 0) & (df['danceability'] <= 1) &
    (df['energy'] >= 0) & (df['energy'] <= 1) &
    (df['instrumentalness'] >= 0) & (df['instrumentalness'] <= 1) &
    (df['liveness'] >= 0) & (df['liveness'] <= 1) &
    (df['speechiness'] >= 0) & (df['speechiness'] <= 1) &
    (df['valence'] >= 0) & (df['valence'] <= 1)

]

durations = df[df['duration_ms'] >= 0]
durmean = durations['duration_ms'].mean()
df['duration_ms'].mask(df['duration_ms'] == -1, durmean, inplace=True)

df['mode'] = df['mode'].replace({'Minor': 0, 'Major': 1})
df['mode'] = df['mode'].astype(int)

df['music_genre'] = df['music_genre'].replace(
    {
        'Electronic': 0, 
        'Alternative': 1,
        'Anime': 2,
        'Jazz': 3,
        'Country': 4,
        'Rap': 5,
        'Blues': 6,
        'Rock': 7,
        'Classical': 8,
        'Hip-Hop': 9
    }
)
df['music_genre'] = df['music_genre'].astype(int)


# In[18]:


for i in range(10):
    tempchangedf = df[(df['music_genre'] == i) & (df['tempo'] != "?")]
    tempchangedf['tempo'] = tempchangedf['tempo'].astype(float)
    avg = tempchangedf['tempo'].mean()
    # Condition on column A and B
    condition = (df['tempo'] == "?") & (df['music_genre'] == i)
    # Replace values in column 'A' where both conditions are met
    df.loc[condition, 'tempo'] = avg
    
df['tempo'] = df['tempo'].astype(float)


# In[19]:


test_dfs = []
train_dfs = []

for i in range(10):
    genre_data = df[df['music_genre'] == i]
    test_df = genre_data.sample(n=500, random_state=42)  # Test set sample
    train_df = genre_data.drop(test_df.index).sample(n=4500, random_state=42)  # Training set sample
    test_dfs.append(test_df)
    train_dfs.append(train_df)
    
trainset = pd.concat(train_dfs)
testset = pd.concat(test_dfs)

preds = [
    'popularity','acousticness','danceability','duration_ms','energy','instrumentalness',
    'liveness','loudness','mode','speechiness','tempo','valence','key_A', 'key_A#', 'key_B', 
    'key_C', 'key_C#', 'key_D', 'key_D#', 'key_E', 'key_F', 'key_F#', 'key_G', 'key_G#'
]

xTrain = trainset[preds]
yTrain = trainset['music_genre']
xTest = testset[preds]
yTest = testset['music_genre']


# In[20]:


scaler = StandardScaler()
scaledXtrain = scaler.fit_transform(xTrain)
scaledXtest = scaler.transform(xTest)


# In[21]:


pca = PCA()
xTrainPCA = pca.fit_transform(scaledXtrain, yTrain)
xTestPCA = pca.transform(scaledXtest)
print("Eigenvalues:", pca.explained_variance_)


# In[22]:


lda = LDA(n_components=9)
xTrainLDA = lda.fit_transform(scaledXtrain, yTrain)
xTestLDA = lda.transform(scaledXtest)


# In[23]:


bdt = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2, class_weight='balanced'), algorithm="SAMME", n_estimators=50, learning_rate=1
)

bdt.fit(xTrainLDA, yTrain)

# yTest = np.argmax(yTest, axis=0)

yPred = bdt.predict(xTestLDA)
yProb = bdt.predict_proba(xTestLDA)
roc_score = roc_auc_score(yTest, yProb, multi_class='ovr')   

print(f"ROC AUC for full adaBoost model:", roc_score)
print("---------------------------------------------------------------------------")


# In[24]:


false_positive_rate, true_positive_rate, threshold = roc_curve(yTest, yProb[:,0], pos_label=yTest.unique()[0])
plt.subplots(1, figsize=(10,10))
plt.title(f'ROC Curve for adaBoost')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[25]:


lda = LDA(n_components=2)
xTrainLDA2 = lda.fit_transform(scaledXtrain, yTrain)

# Apply KMeans with the optimal number of clusters
kmeans = KMeans(n_clusters=10, n_init='auto')
labels = kmeans.fit_predict(xTrainLDA2)

# Plot the clustered data
plt.figure(figsize=(8, 6))
plt.scatter(xTrainLDA2[:, 0], xTrainLDA2[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.5)
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('Clustered Data in 2D LDA Space')
plt.grid(True)
plt.colorbar(label='Cluster ID')
plt.show()


# In[ ]:




