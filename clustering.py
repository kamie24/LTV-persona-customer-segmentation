#!/usr/bin/env python
# coding: utf-8

# ## Data Exploration and Machine Learning
# 
# _Segmentation/Profiling & Classification Case Study_
# 
# ***
# 
# ## Table of Contents
# 
# * [Introduction](#intro)
# * [General Outline](#outline)
# * [Import Required Libraries](#import)
# * [Load Data](#load)
# * [Explore Data - EDA](#eda)
#     * [Check Data](#check)
#     * [Select Features](#select)
#     * [Explore Categorical Features](#cat-eda)
#     * [Explore Numerical Features](#num-eda)
# * [Group Data](#binning)
# * [Select Features After EDA](#select2)
# * [Clustering](#clustering)
#     * [Select Segmentation Variables](#select-segments)
#     * [Method and Number of Segments](#method)
#     * [Prepare Data](#prepare)
#         * [UMAP Embedding](#umap)
#         * [Compute the Hopkins Statistics](#h-stat)
#     * [Apply K-Means Clustering](#k-means)
#         * [The Elbow Method](#elbow)
#         * [Fit K-Means with Optimal Clusters](#fit-kmeans)
#         * [Confirm the Optimal Clusters with Hierarchical Clustering](#clusters)
#     * [Apply K-Medoids Clustering](#k-medoids)
#         * [Fit K-Medoids with Optimal Clusters](#fit-kmedoids)
#     * [Apply K-Prototypes Clustering](#k-prot)
#         * [Fit K-Prototypes with Optimal Clusters](#fit-kprot)
#     * [Compare the Methods Visually](#compare)
#         * [Check Visually K-Means](#check-vis-kmeans)
#         * [Check Visually K-Medoids](#check-vis-kmedoids)
#         * [Check Visually K-Prototypes](#check-vis-kprot)
# * [Classify & Evaluate Clusters](#classify)
#     * [Classify & Evaluate K-Means Clusters](#classify-kmeans)
#         * [Check Feature Importances for K-Means Clusters](#kmeans-importances)
#     * [Classify & Evaluate K-Medoids Clusters](#classify-kmedoids)
#         * [Check Feature Importances for K-Medoids Clusters](#kmedoids-importances)
#     * [Classify & Evaluate K-Prototypes Clusters](#classify-kprot)
#         * [Check Feature Importances for K-Prototypes Clusters](#kprot-importances)
# * [Group Customers](#segmentation)
#     * [K-Means Clusters](#k-means-clusters)
#     * [Export K-Means Clusters](#export-k-means)
#     * [EDA of K-Means Clusters](#eda-k-means)
#     * [K-Medoids Clusters](#k-medoids-clusters)
#     * [Export K-Medoids Clusters](#export-k-medoids)
#     * [EDA of K-Medoids Clusters](#eda-k-medoids)
#     * [K-Prototypes Clusters](#k-prototypes-clusters)
#     * [Export K-Prototypes Clusters](#export-k-prototypes)
#     * [EDA of K-Prototypes Clusters](#eda-k-prototypes)
# 
# 
# Investor personas are research-based archetypal (modeled) representations of who investors are, what they are trying to accomplish, what goals drive their behavior, how they think, how they buy, and why they make buying decisions. We strive to discover where investors invest including when they decide to close on an investment.
# 
# In Data Analytics, we we may want to organize large sets of data in a few clusters with similar observations within each cluster. **Cluster analysis** is a class of techniques that are used to classify objects or cases into relative groups called clusters. Cluster analysis is also called classification analysis or numerical taxonomy. In cluster analysis, there is no prior information about the group or cluster membership for any of the objects.
# 
# In the case of customer data, customers may only belong to a few segments: customers are similar within each segment but different across segments altogether. We may often want to analyze each segment separately as they may behave differently (e.g. different market segments may have different product preferences and behavioral patterns).
# 
# Cluster analysis is used in a variety of applications. For example, it can be used to identify competitive sets of products, or groups of assets whose prices comove, or for geo-demographic segmentation, etc. In general, it is often necessary to split our data into segments and perform any subsequent analysis within each segment in order to develop (potentially more refined) segment-specific insights. This may be the case even if there are no intuitively “natural” segments in our data.
# 
# 
# We will examine 2K+ customer records (`Multiply` stage investors). We want to conduct a comprehensive assessment of the customer base, which is to include a general overview of the customer base and the development of sound customer segmentation:
# 
# 1. Load data
# 2. Explore data
# 3. Clean and preprocess data
# 4. Cluster data
# 5. Evaluate clusters
# 6. Classify
# 



# Import the required libraries.
import pandas as pd
import numpy as np
from random import sample
from numpy.random import uniform
from math import isnan

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import sys
# !{sys.executable} -m pip install  scikit-learn-extra
from sklearn_extra.cluster import KMedoids
from sklearn.model_selection import cross_val_score
# 
# !{sys.executable} -m pip install yellowbrick
from yellowbrick.cluster.elbow import kelbow_visualizer
# !{sys.executable} -m pip uninstall  umap
# !{sys.executable} -m pip install  umap-learn
# import umap.umap_ as umap

import scipy.cluster.hierarchy as sch

from kmodes.kprototypes import KPrototypes

from lightgbm import LGBMClassifier


# import shap

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# Set Plotly theme.
pio.templates.default = "gridon"

# Set global variables.
RANDOM_STATE = 5 # set a seed so as to have reproducibility in the analysis.

get_ipython().run_line_magic('matplotlib', 'inline')


  
# 
# 
# Read the dataset from CSV and load it into a Pandas dataframe.



# Read data.
df = pd.read_csv('data-to-cluster-updated.csv')

# Display some info.

# Show first rows.


  
# 
# 
# The Exploratory Data Analysis or EDA include the following steps:
# 
# * Review the available data and select specific variables of interest.
# * Check the quality of data.
# * Check for imbalances and create charts.
# * Identify opportunities, if any, to recode current variables or create new ones combining variables into a single measure.
# 
# 
# The quality of the dataframe will be examined. The types and shapes of the data itself will be analyzed, including any missing or duplicated records.



# Create a function to check the data.
def check_data(df): 
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()]).T.squeeze()
    duplicates = df.duplicated().sum()
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing_ratio = round((df.isnull().sum()/ obs) * 100, 2)
    skewness = df.skew()
    kurtosis = df.kurt() 
    print('Data shape: ', df.shape)
    print('Duplicates: ', duplicates)
    frame = {'types': types, 'counts': counts, 'uniques': uniques, 'nulls': nulls, 'distincts': distincts,
             'missing_ratio': missing_ratio, 'skewness': skewness, 'kurtosis': kurtosis}
    checks = pd.DataFrame(frame)

check_data(df)


  
# 
# Select and group the features to:
# 
# * Profile (demographic/behavioural) attributes and
# * Segmentation attributes

# In[13]:


# Keep only the columns of interest.
df = df[['id', 'count_of_investments', 'sum_of_investments',
         'users_age', 'occupation_normalized',
         'investment_objectives', 'ideal_re_investment_amount_per_deal',
         'ideal_re_investment_amount_per_year', 'entity_type',
         'investment_experience', 'annual_income',
         'risk_tolerance', 'years_investing_illiquid',
         'marital_status', 'investment_time_horizon', 'liquid_net_worth',
         'est_net_worth', 'years_investing_securities',
         'preferred_asset_classes', 'preferred_investment_type',
         'preferred_market_types', 'preferred_property_types',
         'target_cash_on_cash_return', 'target_irr', 'top_investment_criteria',
         'income_information_type', 'investment_goals', 'investment_range',
         'investment_style']]

# Select the numerical features.
num_feats = ['users_age', 'count_of_investments', 'sum_of_investments']

# Select the categorical features.
cat_feats = [ele for ele in df.columns if ele not in num_feats]
cat_feats.remove('id')

# Fill NaNs with 'Unknown'.
df.fillna('Unknown', inplace = True)


  
# 
# 
# Group the categorical features and plot the counts for each category.



for feat in cat_feats:
    # Show counts for each feature.
    fig = px.bar(df.groupby(feat).count().reset_index().sort_values(by=['id']),
                 x='id', y=feat, text='id',
                 opacity=0.6, orientation='h')
    fig.update_layout(title_text="Distribution of " + feat)
    fig.update_xaxes(showgrid=False, title_text=None)
    fig.update_yaxes(showgrid=False, title_text=None)
    fig.update_yaxes(showticklabels=True, automargin=True)
    fig.show()


   
# 
# * Here, we can see the most common categories of each feature.
# * `Unknown` values are actually missing values (nulls) which dominate most features in the dataset. Let's remove these features, since they are not very helpful.
# 
  
# 
# 
# Plot combined histogram-box or violin charts for the numerical features.

# In[15]:


for feat in num_feats:
    fig = px.histogram(df, x=feat, marginal="violin") # or box, rug for marginal
    fig.show()


   
# 
# * We can see many outliers for most features.
# * Maybe we should choose K-medoids later as the clustering algorithm. The K-means clustering algorithm is sensitive to outliers, because a mean is easily influenced by extreme values, while K-medoids clustering is a variant of K-means that is more robust to noises and outliers.
# * Alternatively, we could log/power transform the numerical data, so as to become more Gauss distribution like (bell-shaped curve that has the assumption that during any measurement values will follow a normal distribution with an equal number of measurements above and below the mean value) or group them into bins.
# 
  
# 
# 
# **Inbalance Issues:** Pandas provides the `pandas.cut` tool that allows us to divide data by user-defined bins, while `pandas.qcut` can create quantile-based discretization (the process of transferring continuous functions, models, variables, and equations into discrete counterparts). In quintiles with qcut, the bins will be chosen so that we get the same number of records in each bin. This approach takes care of imbalance issues. The simplest use of qcut is to define the number of quantiles and let Pandas figure out how to divide up the data.



# Group data into equal-sized buckets.
# df['income_expected_for_current_year'] = pd.qcut(df['income_expected_for_current_year'],
#                                                  q=4, precision=0).astype('str')
# df['count_of_investments'] = pd.qcut(df['count_of_investments'],
#                                      q=5, precision=0, duplicates='drop').astype('str')
# df['sum_of_investments'] = pd.qcut(df['sum_of_investments'],
#                                    q=5, precision=0, duplicates='drop').astype('str')

# Show first rows




# Keep only the columns of interest.
df = df[['id', 'count_of_investments', 'sum_of_investments',
         'users_age', 'occupation_normalized', 'investment_objectives',
         'investment_experience', 'risk_tolerance', 'years_investing_illiquid',
         'marital_status', 'investment_time_horizon', 'liquid_net_worth',
         'years_investing_securities', 'income_information_type'
        ]]

# Select the numerical features.
num_feats = ['users_age', 'count_of_investments', 'sum_of_investments']

# Select the categorical features.
cat_feats = [ele for ele in df.columns if ele not in num_feats]
cat_feats.remove('id')

# Fill NaNs with 'Unknown'.
df.fillna('Unknown', inplace = True)


  
# 
# Customer segmentation is the process of dividing an organization’s customer bases into different sections based on various customer attributes. The process of customer segmentation is based on the premise of finding differences among the customers’ behavior and patterns.
# 
# The major objectives and benefits behind the motivation for customer segmentation are:
# 
# * Increase revenue
# * Target marketing intiatives with granularity
# * Understand the customer in detail
# * Find latent customer segments (i.e. find out which segment of customers it might be missing and apply new marketing campaigns)
# * Develop a strategy that can offer new products or a bundle of products together as a combined offering (optimal product placement).
# 
# **Clustering:**
# 
# Common methods to perform customer segmentation are the unsupervised ML methods like clustering. The method is as simple as collecting as much data about the customers as possible in the form of features or attributes and then finding out the different clusters that can be obtained from that data.
# 
# The decision about which variables to use for clustering is a critically important decision that will have a big impact on the clustering solution. Sound exploratory research that provides a good sense of what variables may distinguish is critical. This is a step where a lot of contextual knowledge, creativity, and experimentation / iterations are needed.
# 
# Moreover, we often use only a select few data attributes for segmentation and use some of the remaining attributes only to profile the clusters. For example, in market research and market segmentation, one may use attitudinal data for segmentation (to segment the customers based on their needs and attitudes towards the products / services) and then demographic and behavioral data for profiling the segments found therein.
# 
# There are many statistical methods for clustering and segmentation. In this study, we will use 3 widely used methods: the **K-Means** or **K-Medoids** clustering method, the **Hierarchical Clustering** method, and the **K-Prototypes** method. K-Means and K-Prototypes methods require the user to define how many segments to create, while Hierarchical Clustering does not.
# 
# Since we do not know for now how many segments there are in our data, we can use the Hierarchial Clustering as an evaluation method. Hierarchical clustering is a method that also helps us visualise how the data may be clustered together. It generates a plot called the **Dendrogram**, which is helpful for visualization.
# 
# The Dendrogram indicates how this clustering method works: observations are “grouped together”, starting from pairs of individual observations which are the closest to each other, and merging smaller groups into larger ones depending on which groups are closest to each other. Eventually all the data are merged into one segment. The heights of the branches of the tree indicate how different the clusters merged at that level of the tree are. Longer lines indicate that the clusters below are very different. As expected, the heights of the tree branches increase as we traverse the tree from the end leaves to the tree root: the method merges data points/groups from the closest ones to the furthest ones.
# 
# Some of the methods that are used to evaluate the optimal number of clusters are:
# 
# * **Elbow Method**
# * **Hierarchical Clustering**
# * **Silhouette Coefficient & Silhouette Analysis Charts**
# * **UMAP Embedding**
# * Evaluate the clusters fitting a **Classification Algorithm**
# 
# Our dataset is a mix of numerical and categorical data. The standard K-Means algorithm isn't directly applicable to categorical data, for various reasons. The sample space for categorical data is discrete, and doesn't have a natural origin. A Euclidean distance function on such a space isn't really meaningful.
# 
# There's a variation of K-Means known as **K-Modes**, introduced by Zhexue Huang, which is suitable for categorical data. Huang's paper has also a section on **K-Prototypes**, which applies to data with a mix of categorical and numerical features. K-Prototypes offers the advantage of workign with mixed data types. It measures distance between numerical features using **Euclidean distance** (like K-Means), but also measure the distance between categorical features using the **Hamming distance**.
# 
# One of the evaluation methods is visual using the **Uniform Manifold Approximation and Projection for Dimension Reduction (UMAP)** method - a dimensionality reduction technique (like PCA or t-SNE) - to embedd the data into 2 dimensions. This allows to see the groups of customers and how well did the clustering algorithms do the job. There are 3 steps to get the proper embeddings:
# 
# * **Yeo-Johnson** transformation of the numerical columns & **One-Hot-Encode** of the categorical features. The Yeo-Johnson method is a useful data transformation technique used to stabilize variance, make the data more normal distribution-like, improve the validity of measures of association such as the Pearson correlation between variables and for other data stabilization procedures.
# * Embed these two columns types separately.
# * Combine the two by conditioning the numerical embeddings on the categorical embeddings.

 


# Set index.
# !{sys.executable} -m pip uninstall umap
# !{sys.executable} -m pip install umap-learn
# import umap.umap_ as umap

df.set_index('id', inplace=True)

# Preprocess numerical.
numerical = df.select_dtypes(exclude='object')

for c in numerical.columns:
    pt = PowerTransformer()
    numerical.loc[:, c] = pt.fit_transform(np.array(numerical[c]).reshape(-1, 1))
    
# Preprocess categorical.
categorical = df.select_dtypes(include='object')
categorical = pd.get_dummies(categorical)

# Use percentage of columns, which are categorical as a weight parameter in embeddings later.
categorical_weight = len(df.select_dtypes(include='object').columns) / df.shape[1]

# Embed numerical & categorical.
fit1 = umap.UMAP(metric='l2', random_state=RANDOM_STATE).fit(numerical)
fit2 = umap.UMAP(metric='dice', random_state=RANDOM_STATE).fit(categorical)

# See the categorical_weight.
categorical_weight


 


# Augment the numerical embedding with categorical.
intersection = umap.umap_.general_simplicial_set_intersection(fit1.graph_, fit2.graph_, weight=categorical_weight)
intersection = umap.umap_.reset_local_connectivity(intersection)
embedding = umap.umap_.simplicial_set_embedding(fit1._raw_data, intersection, fit1.n_components, 
                                                fit1._initial_alpha, fit1._a, fit1._b, 
                                                fit1.repulsion_strength, fit1.negative_sample_rate, 
                                                200, 'random', np.random, fit1.metric, 
                                                fit1._metric_kwds, True)

# Plot.
fig = px.scatter(x=embedding.T[0], y=embedding.T[1], opacity=0.5)
fig.show()


   
# 
# * To find out the actual number of custers, we will use various methods.
# 
  
# 
# To find if the dataset can be clustered, we can use the **Hopkins Statistic**, which tests the spatial randomness of the data and indicates the cluster tendency or how well the data can be clustered. It calculates the probability that a given data is generated by a uniform distribution. The inference is as follows for a data of dimensions '*d*':
# 
# * If the value is around 0.5 or lesser, the data is uniformly distributed and hence it is unlikely to have statistically significant clusters.
# * If the value is between {0.7, ..., 0.99}, there is a high tendency to cluster and therefore likely to have statistically significant clusters.

 


# Create a function to compute Hopkins Statistic as a way of measuring the cluster tendency of a data set.
def compute_hopkins(X):
    d = X.shape[1]
    n = len(X)
    m = int(0.1 * n) # heuristic
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),
                                            np.amax(X,axis=0),d).reshape(1, -1), 2,
                                    return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2,
                                    return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H


 


# Check whether data can be clustered. It can be applied only on numerical features.
compute_hopkins(numerical)


   
# 
# * Running the test on all the numerical variables of the entire dataset, we get a very high h-statistic, which indicates that the data has a high tendency to cluster - at least the numerical features.
# * Now, let's begin actual modelling with K-Means, K-Medoids, and K-Prototypes. There are also other models like DBSCAN Clustering, but this works only with numerical variables by aggregation depending on density of points near the chosen centroids.
# 
  
# 
# Because K-Means only works with numerical data, we need to:
# 
# * One-Hot-Encode the categorical data.
# * Apply the Yeo-Johnson transformation to make numerical features it more Gaussian like.
# * Fit K-means and use the Elbow Method to find the optimal number of clusters.
# 
# This is a technique that is used to help us find the optimal number of clusters. This method looks at the percentage of variance explained as a function of the number of clusters: One should choose a number of clusters so that adding another cluster doesn't give much better modeling of the data. More precisely, if one plots the percentage of variance explained by the clusters against the number of clusters, the first clusters will add much information (explain a lot of variance), but at some point the marginal gain will drop, giving an angle in the graph. The number of clusters is chosen at this point, hence the "elbow criterion". This "elbow" cannot always be unambiguously identified.

 


# One-Hot-Encode.
data = pd.get_dummies(df)

# Yeo-Johnson transform.
for c in data.columns:
    pt = PowerTransformer()
    data.loc[:, c] = pt.fit_transform(np.array(data[c]).reshape(-1, 1))


 


# Apply the elbow method to find the optimal number of clusters.
kelbow_visualizer(KMeans(init='k-means++', max_iter=1000, random_state=RANDOM_STATE), data, k=(2, 50))


 


# Apply the silhouette method to find the optimal number of clusters.
kelbow_visualizer(KMeans(init='k-means++', max_iter=1000, random_state=RANDOM_STATE),
                  data, k=(2, 50), metric='silhouette')


   
# 
# * Too many clusters from the Elbow method - most of the groups contain few customers.
# * Let's keep it simple and use Silhouette score heuristics.
# 
  
# Fit K-Means with the optimal number of clusters.

 


kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=1000, random_state=RANDOM_STATE).fit(data)
kmeans_labels = kmeans.labels_


  
# Let's confirm the optimal number of clusters with another statistical technique called **Hierarchical Clustering**.

 


# Use the dendrogram to find the optimal number of clusters.
plt.figure(figsize=(20, 10))
dendrogram = sch.dendrogram(sch.linkage(data, method = 'ward')) # 'Ward' minimizes the within-cluster variance.
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


   
# 
# * The results are too complicated, thus making the results inconclusive.
# 
  
# 
# 
# Let's use K-Medoids, which is an alternative technique to centroid-based method for partitional clustering algorithms. This method is more robust for categorical data and data with outliers.

 


# Apply the elbow method to find the optimal number of clusters.
kelbow_visualizer(KMedoids(metric="manhattan", max_iter=1000, random_state=RANDOM_STATE), data, k=(2, 50))


 


# Apply the silhouette method to find the optimal number of clusters.
kelbow_visualizer(KMedoids(metric="manhattan", max_iter=1000, random_state=RANDOM_STATE),
                  data, k=(2, 50), metric='silhouette')


   
# 
# * Both Elbow and Silhouette methods give the same optimal number of clusters.
# 
  
# 
# 
# Fit K-Medoids with the optimal number of clusters.

 


kmedoids = KMedoids(metric="manhattan", n_clusters=11, max_iter=1000, random_state=RANDOM_STATE).fit(data)
kmedoids_labels = kmedoids.labels_


  
# 
# 
# This model can deal with numerical and categorical data, so we don't need to one-hot-encode the categorical features. We only need to apply the Yeo-Johnson transformation to the numerical data.

 


# Create a copy of the initial df.
kprot_data = df.copy()

# Transform the numerical features.
for c in df.select_dtypes(exclude='object').columns:
    pt = PowerTransformer()
    kprot_data[c] = pt.fit_transform(np.array(kprot_data[c]).reshape(-1, 1))


 


# In KPrototypes, we need to indicate which column indices are categorical using the categorical argument.
# All others are assumed numerical.
categorical_columns = df.columns.get_indexer(cat_feats).tolist()


 


costs = []
n_clusters = []
clusters_assigned = []

for k in tqdm(range(2, 20)):
    try:
        kproto = KPrototypes(n_clusters=k, init='Cao', n_jobs=-1, random_state=RANDOM_STATE)
        clusters = kproto.fit_predict(kprot_data, categorical=categorical_columns)
        costs.append(kproto.cost_)
        n_clusters.append(k)
        clusters_assigned.append(clusters)
    except:
        print(f"Can't cluster with {k} clusters")
        
fig = go.Figure(data=go.Scatter(x=n_clusters, y=costs))
fig.update_xaxes(title_text='Number of clusters')
fig.update_yaxes(title_text='Cost')
fig.update_layout(title = "The Elbow Method in K-Prototypes Clustering")
fig.show()


   
# 
# * We can say that we have an "elbow" at about 5 clusters.
# 
  
# 
# 
# Fit K-Prototypes with the optimal number of clusters.

 


kproto = KPrototypes(n_clusters=5, init='Cao', n_jobs=-1, random_state=RANDOM_STATE)
kproto_clusters = kproto.fit_predict(kprot_data, categorical=categorical_columns)


  
# 
# 
# We can evaluate visually the sets of clusters by colouring the dots of the UMAP embeddings from above and see which makes more sense.
# 

 


fig, ax = plt.subplots()
fig.set_size_inches((20, 10))
scatter = ax.scatter(embedding[:, 0], embedding[:, 1], s=10, c=kmeans_labels, cmap='tab20b', alpha=0.5)

# Produce a legend with the unique colors from the scatter.
legend1 = ax.legend(*scatter.legend_elements(num=6),
                    loc="lower left", title="Classes")
ax.add_artist(legend1)


   
# 
# * The classes are not very clear.
# 
  
# 

 


fig, ax = plt.subplots()
fig.set_size_inches((20, 10))
scatter = ax.scatter(embedding[:, 0], embedding[:, 1], s=10, c=kmedoids_labels, cmap='tab20b', alpha=0.5)

# Produce a legend with the unique colors from the scatter.
legend1 = ax.legend(*scatter.legend_elements(num=3),
                    loc="lower left", title="Classes")
ax.add_artist(legend1)


   
# 
# * The colours between the groups are a bit more clear with this model.
# 
  
# 

 


fig, ax = plt.subplots()
fig.set_size_inches((20, 10))
scatter = ax.scatter(embedding[:, 0], embedding[:, 1], s=10, c=kproto_clusters, cmap='tab20b', alpha=0.5)

# Produce a legend with the unique colors from the scatter.
legend1 = ax.legend(*scatter.legend_elements(num=4),
                    loc="lower left", title="Classes")
ax.add_artist(legend1)


   
# 
# * The classes look more distinct with this model.
# 
  
# 
# 
# Another comparison that we can apply is treating the clusters as *labels* and building a classification model on top of them. If the clusters are of high quality, the classification model will be able to predict the clusters with high accuracy. At the same time, the models should use a variety of features to ensure that the clusters are not too simplistic.
# 
# We will validate and evaluate the quality of clusters by:
# 
# * The cross-validated F1 score
# * The informativness of clusters by **SHAP feature importances**
# 
# We will use the `LightGBM` as a classification model, since it can use categorical features and can easily provide the SHAP values for the trained models.

 


# Create a copy of the initial df.
lgbm_data = df.copy()

# Set the objects to category.
for c in lgbm_data.select_dtypes(include='object'):
    lgbm_data[c] = lgbm_data[c].astype('category')


 


# Classify K-Means clusters.
clf_km = LGBMClassifier(colsample_by_tree=0.8, random_state=RANDOM_STATE)
cv_scores_km = cross_val_score(clf_km, lgbm_data, kmeans_labels, scoring='f1_weighted')
print('The Cross-Validated F1-Weighted score for K-Means clusters is {}'.format(np.mean(cv_scores_km)))


   
# 
# * We achieved a very high score, meaning that the customers are grouped in meaningful and distinguishable clusters.
# 
  
# 
# Let’s also investigate the importances of the features to check if the classifier used all the information available to it. The bar chart below shows the mean absolute value of the SHAP values for each feature.

 


clf_km.fit(lgbm_data, kmeans_labels)

explainer_km = shap.TreeExplainer(clf_km)
shap_values_km = explainer_km.shap_values(lgbm_data)

shap.summary_plot(shap_values_km, lgbm_data, plot_type="bar", plot_size=(15, 10))


   
# 
# * It seems that the classifier has used many features. This means that the model is quite informative.
# 
  
# 
# 
# Let’s apply the same methodology to K-Medoids clusters to check if the classifier can use other features in grouping the customers.

 


# Create a copy of the initial df.
lgbm_data = df.copy()

# Set the objects to category.
for c in lgbm_data.select_dtypes(include='object'):
    lgbm_data[c] = lgbm_data[c].astype('category')


 


# Classify K-Medoids clusters.
clf_kmed = LGBMClassifier(colsample_by_tree=0.8, random_state=RANDOM_STATE)
cv_scores_km = cross_val_score(clf_kmed, lgbm_data, kmedoids_labels, scoring='f1_weighted')
print('The Cross-Validated F1-Weighted score for K-Medoids clusters is {}'.format(np.mean(cv_scores_km)))


   
# 
# * We get a lower score than K-Means.
# 
  
# 
# 
# Let’s also investigate the feature importances to check if the classifier used all the information available to it.

 


clf_kmed.fit(lgbm_data, kmedoids_labels)

explainer_kmed = shap.TreeExplainer(clf_kmed)
shap_values_kmed = explainer_kmed.shap_values(lgbm_data)

shap.summary_plot(shap_values_kmed, lgbm_data, plot_type="bar", plot_size=(15, 10))


   
# 
# * It seems that the classifier has many features.
# 
  
# 
# 
# Let’s apply the same methodology to K-Prototypes clusters to check if the classifier can use other features in grouping the customers.

 


# Classify K-Means clusters.
clf_kp = LGBMClassifier(colsample_by_tree=0.8, random_state=RANDOM_STATE)
cv_scores_kp = cross_val_score(clf_kp, lgbm_data, kproto_clusters, scoring='f1_weighted')
print('The Cross-Validated F1-Weighted score for K-Prototypes clusters is {}'.format(np.mean(cv_scores_kp)))


   
# 
# * We get a good score similar to K-Means.
# 
  
# 
# 
# See the feature importances to check if the classifier used all the information available to it.

 


clf_kp.fit(lgbm_data, kproto_clusters)

explainer_kp = shap.TreeExplainer(clf_kp)
shap_values_kp = explainer_kp.shap_values(lgbm_data)

shap.summary_plot(shap_values_kp, lgbm_data, plot_type="bar", plot_size=(15, 10))


   
# 
# * Just 2 features are by far the most important.
# * We can say that the clusters produced by K-Prototypes are less informative.
# 
  
# 
# Let's see now the number of segments and the number of customers in each segment in more detail. First, we need to assign every customer to the clusters. We will use the results from both K-Means and K-Prototypes.
# 

 


# Create a column for the clusters.
lgbm_data = lgbm_data.assign(Cluster=pd.Series(kmeans_labels).values)


 


# Show counts for each cluster.
fig = px.bar(lgbm_data.groupby('Cluster').count().reset_index().sort_values(by=['Cluster']),
             x='Cluster', y=feat, text='Cluster',opacity=0.6)
fig.update_layout(title_text="Distribution of K-Means Clusters")
fig.update_xaxes(showgrid=False, title_text="Clusters")
fig.update_yaxes(showgrid=False, title_text="Number of Customers")
fig.update_yaxes(showticklabels=True, automargin=True)
fig.show()

# Show descriptive stats for each cluster.
lgbm_data_stats = lgbm_data.groupby(by=['Cluster']).describe(include='all').T


   
# 
# * The bar plot shows that there is 2 major clusters with about 1000 customers each and 2 smaller ones with about 150-200 customers each.
# 
  
# 
# Export the datasets.

 


lgbm_data.to_csv("k-means-clusters.csv")
lgbm_data_stats.to_csv("k-means-clusters-stats.csv")


  
# 
# Do a simple EDA of the K-Means data.

 


# Make a copy of the initial data.
eda_kmeans = df.copy().reset_index()

# One-hot-encode the categorical features.
eda_kmeans = pd.get_dummies(eda_kmeans)

# Add the clusters to the data.
eda_kmeans['Cluster'] = kmeans_labels

# Unpivot from wide to long format.
eda_kmeans = eda_kmeans.melt(id_vars=['id', 'Cluster'])

# Remove the useless "id" column.
eda_kmeans.drop("id", axis=1, inplace=True)

g = sns.FacetGrid(eda_kmeans.groupby(['Cluster', 'variable']).mean().reset_index(),
                  col='variable', hue='Cluster', col_wrap=14, height=5, sharey=False)
g = g.map(plt.bar, 'Cluster', 'value').set_titles("{col_name}")


   
# 
# * There are too many variables to check, but if we see carefully, we maybe identify distinct segments with clear business implications.
# 
  
 


# Create a column for the clusters.
lgbm_data = lgbm_data.assign(Cluster=pd.Series(kmedoids_labels).values)


 


# Show counts for each cluster.
fig = px.bar(lgbm_data.groupby('Cluster').count().reset_index().sort_values(by=['Cluster']),
             x='Cluster', y=feat, text='Cluster',opacity=0.6)
fig.update_layout(title_text="Distribution of K-Medoids Clusters", yaxis={'tickformat': ',d'})
fig.update_xaxes(showgrid=False, title_text="Clusters")
fig.update_yaxes(showgrid=False, title_text="Number of Customers")
fig.update_yaxes(showticklabels=True, automargin=True)
fig.show()

# Show descriptive stats for each cluster.
lgbm_data_stats = lgbm_data.groupby(by=['Cluster']).describe(include='all').T


   
# 
# * The bar plot shows that there is 4 major clusters with about 250-600 customers each.
# 
# Export the datasets.

 


lgbm_data.to_csv("k-medoids-clusters.csv")
lgbm_data_stats.to_csv("k-medoids-clusters-stats.csv")


  
# Do a simple EDA of the K-Medoids data.

 


# Make a copy of the initial data.
eda_kmedoids = df.copy().reset_index()

# One-hot-encode the categorical features.
eda_kmedoids = pd.get_dummies(eda_kmedoids)

# Add the clusters to the data.
eda_kmedoids['Cluster'] = kmedoids_labels

# Unpivot from wide to long format.
eda_kmedoids = eda_kmedoids.melt(id_vars=['id', 'Cluster'])

# Remove the useless "id" column.
eda_kmedoids.drop("id", axis=1, inplace=True)

g = sns.FacetGrid(eda_kmedoids.groupby(['Cluster', 'variable']).mean().reset_index(),
                  col='variable', hue='Cluster', col_wrap=14, height=5, sharey=False)
g = g.map(plt.bar, 'Cluster', 'value').set_titles("{col_name}")


   
# * There are too many variables to check, but if we see carefully, we maybe identify distinct segments with clear business implications.
  

 


# Create a column for the clusters.
lgbm_data = lgbm_data.assign(Cluster=pd.Series(kproto_clusters).values)


 


# Show counts for each cluster.
fig = px.bar(lgbm_data.groupby('Cluster').count().reset_index().sort_values(by=['Cluster']),
             x='Cluster', y=feat, text='Cluster',opacity=0.6)
fig.update_layout(title_text="Distribution of K-Prototypes Clusters")
fig.update_xaxes(showgrid=False, title_text="Clusters")
fig.update_yaxes(showgrid=False, title_text="Number of Customers")
fig.update_yaxes(showticklabels=True, automargin=True)
fig.show()

# Show descriptive stats for each cluster.
lgbm_data_stats = lgbm_data.groupby(by=['Cluster']).describe(include='all').T


   
# * There are 5 clusters with many users each.
# Export the datasets.

 


lgbm_data.to_csv("k-prototypes-clusters.csv")
lgbm_data_stats.to_csv("k-prototypes-clusters-stats.csv")


  
# Do a simple EDA of the K-Prototypes data.

 


# Make a copy of the initial data.
eda_kprot = df.copy().reset_index()

# One-hot-encode the categorical features.
eda_kprot = pd.get_dummies(eda_kprot)

# Add the clusters to the data.
eda_kprot['Cluster'] = kproto_clusters

# Unpivot from wide to long format.
eda_kprot = eda_kprot.melt(id_vars=['id', 'Cluster'])

# Remove the useless "id" column.
eda_kprot.drop("id", axis=1, inplace=True)

g = sns.FacetGrid(eda_kprot.groupby(['Cluster', 'variable']).mean().reset_index(),
                  col='variable', hue='Cluster', col_wrap=14, height=5, sharey=False)
g = g.map(plt.bar, 'Cluster', 'value').set_titles("{col_name}")


   
# * There are too many variables to check, but if we see carefully, we can identify distinct segments with clear business implications. This set of clusters are the most meaningful and are very similar to what was produced for the persona trend analysis.
  

 




