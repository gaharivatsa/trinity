#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_excel(r"D:\machine learning\trinity\claims_set.xlsx")


# In[3]:


df


# In[4]:


df.describe()


# In[5]:


df.columns


# In[6]:


df.head(10)


# In[7]:


df['patient year of birth']


# In[ ]:





# In[8]:


def calculate_age(year_of_birth):
    if year_of_birth != 0:
        return 2024 - year_of_birth
    else:
        return 0

df['age'] = df['patient year of birth'].apply(calculate_age)


# In[9]:


df['mean over age'] = [(i-df['age'].mean()) for i in df['age']]


# In[10]:


df['mean over age']


# In[11]:


df['product name'].loc[df['product name'] == '\xa0'].size


# In[12]:


def change_to_null(x):
    if x == '\xa0':
        return np.nan
    else:
        return x
df['product name'] = df['product name'].apply(change_to_null)


# In[13]:


df_dropped = df.dropna(subset = ['product name'])


# In[14]:


df_dropped['product name']


# In[33]:


df_dropped.to_excel(r'D:\machine learning\trinity\4_ques.xlsx')


# In[15]:


group_drug = df.groupby('product name')


# In[16]:


avg_age_per_drug = group_drug['patient year of birth'].mean()


# In[17]:


avg_age_per_drug_df = avg_age_per_drug.reset_index(name = 'avg age')


# In[18]:


avg_age_per_drug_df['avg age'] = avg_age_per_drug_df['avg age'].apply(calculate_age)


# 1.1 ANSWER

# In[19]:


avg_age_per_drug_df


# In[20]:


unique_patients = df.groupby(['product name','primary plan payment type'])['patient id'].nunique()


# 1.2

# In[21]:


unique_patients


# In[22]:


avg_age_per_drug_df.to_excel(r'D:\machine learning\trinity\drugs_age.xlsx')


# In[31]:


avg_age_per_drug_df


# In[23]:


import matplotlib.pyplot as plt

# Step 1: Group the data by payment types
grouped_by_payment_type = df.groupby('primary plan payment type')

# Step 2: Group the grouped data by claim date and count claim submissions
claim_submissions_by_date_and_payment_type = grouped_by_payment_type['claim date'].value_counts().unstack(level=0).fillna(0)

# Step 3: Plot the data on a line chart
plt.figure(figsize=(10, 6))
for payment_type in claim_submissions_by_date_and_payment_type.columns:
    plt.plot(claim_submissions_by_date_and_payment_type.index, claim_submissions_by_date_and_payment_type[payment_type], marker='o', label=payment_type)

plt.title('Trend in Claim Submissions Over Time for Each Payment Type')
plt.xlabel('Claim Date')
plt.ylabel('Number of Claim Submissions')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# In[24]:


df['claim date'].size


# In[30]:


import matplotlib.pyplot as plt


hcp_per_drug = df.groupby('product name')['rendering provider hcp id'].nunique()


colors = plt.cm.tab20.colors  

plt.figure(figsize=(10, 8))
explode = [0.1] * len(hcp_per_drug)  
plt.pie(hcp_per_drug, labels=hcp_per_drug.index, autopct='%1.1f%%', startangle=140, colors=colors, explode=explode, shadow=True)
plt.title('Percentage of HCPs Associated with Different Drugs', fontsize=16, fontweight='bold')
plt.axis('equal')  
plt.tight_layout() 
plt.show()


# In[35]:


df_dropped


# In[74]:


hcp = pd.read_excel(r"D:\machine learning\trinity\hcp_data.xlsx")


# In[75]:


hcp.head(5)


# In[76]:


hcp.columns


# In[79]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

# Selecting relevant feature for segmentation
feature = 'Primary  specialty description'

# Dropping rows with missing values in the selected feature
hcp.dropna(subset=[feature], inplace=True)

# One-hot encoding categorical feature
encoder = OneHotEncoder()
encoded_feature = encoder.fit_transform(hcp[[feature]])

# Concatenating the encoded feature with the original dataframe
hcp_encoded = pd.concat([hcp, pd.DataFrame(encoded_feature.toarray(), columns=encoder.get_feature_names_out([feature]))], axis=1)

# Scaling the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(hcp_encoded.drop(columns=[' DoctorID', ' Doctor first name', ' Doctor last name', ' Doctor middle name', 'Secondary specialty code', 'Secondary specilaty description', ' Doctor state', ' Doctor state code', ' Doctor zip', feature]))

# Choosing the number of clusters (you can use methods like elbow method to find optimal k)
k = 3

# Applying K-means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
hcp_encoded['cluster'] = kmeans.fit_predict(scaled_features)

# Visualizing the clusters
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=hcp_encoded['cluster'], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering of Doctors based on Primary Specialty')
plt.colorbar(label='Cluster')
plt.show()


# In[84]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from mpl_toolkits.mplot3d import Axes3D


feature = 'Primary  specialty description'


hcp.dropna(subset=[feature], inplace=True)

encoder = OneHotEncoder()
encoded_feature = encoder.fit_transform(hcp[[feature]])


hcp_encoded = pd.concat([hcp, pd.DataFrame(encoded_feature.toarray(), columns=encoder.get_feature_names_out([feature]))], axis=1)


numerical_cols = hcp_encoded.select_dtypes(include=['float64', 'int64']).columns


imputer = SimpleImputer(strategy='mean')
hcp_encoded[numerical_cols] = imputer.fit_transform(hcp_encoded[numerical_cols])


scaler = StandardScaler()
scaled_features = scaler.fit_transform(hcp_encoded[numerical_cols])


k = 3


kmeans = KMeans(n_clusters=k, random_state=42)
hcp_encoded['cluster'] = kmeans.fit_predict(scaled_features)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


for cluster in hcp_encoded['cluster'].unique():
    cluster_data = scaled_features[hcp_encoded['cluster'] == cluster]
    ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], label=f'Cluster {cluster}')

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('K-means Clustering of Doctors based on Primary Specialty')
ax.legend()

plt.show()


# In[87]:


df4 = pd.read_excel(r"D:\machine learning\trinity\4_ques.xlsx")


# In[89]:


df4.head(10)


# In[92]:


df4.columns


# In[ ]:


county_names = [re.search(r'\((.*?)\)', plan).group(1) if '(' in plan else 'Unknown' for plan in primary_plan_names]

# Count the occurrences of each county
county_distribution = pd.Series(county_names).value_counts()

# Plot the geographic distribution of patients by county
plt.figure(figsize=(10, 6))
county_distribution.plot(kind='bar', color='skyblue')
plt.title('Geographic Distribution of Patients by County')
plt.xlabel('County')
plt.ylabel('Number of Patients')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[94]:


import pandas as pd
import matplotlib.pyplot as plt
import re

# Extract county names from 'primary plan name' column
county_names = df4['primary plan name'].apply(lambda x: re.search(r'\((.*?)\)', x).group(1) if '(' in x else 'Unknown')

# Count the occurrences of each county
county_distribution = county_names.value_counts()

# Plot the geographic distribution of patients by county
plt.figure(figsize=(10, 6))
county_distribution.plot(kind='bar', color='skyblue')
plt.title('Geographic Distribution of Patients by County')
plt.xlabel('County')
plt.ylabel('Number of Patients')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[95]:


import pandas as pd
import matplotlib.pyplot as plt
import re



# Filter relevant columns for analysis
relevant_columns = ['patient id', 'patient year of birth', 'patient gender source value',
                    'claim status', 'primary plan name', 'secondary plan name', 'tertiary plan name',
                    'service bill amount', 'product name']

# Create a DataFrame with relevant columns
df = df4[relevant_columns]

# Calculate patient age
df['patient_age'] = pd.Timestamp.now().year - df['patient year of birth']

# Drop rows with missing values
df.dropna(subset=['patient_age', 'patient gender source value', 'product name'], inplace=True)

# Analyze patient demographics
gender_distribution = df['patient gender source value'].value_counts()
age_distribution = df['patient_age'].value_counts().sort_index()

# Plot patient demographics
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
gender_distribution.plot(kind='bar', color='skyblue')
plt.title('Distribution of Patients by Gender')
plt.xlabel('Gender')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
age_distribution.plot(kind='bar', color='lightgreen')
plt.title('Distribution of Patients by Age')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Analyze geographic distribution
# Extract county names from 'primary plan name' column
county_names = df['primary plan name'].apply(lambda x: re.search(r'\((.*?)\)', x).group(1) if '(' in x else 'Unknown')
county_distribution = county_names.value_counts().head(10)

# Plot geographic distribution
plt.figure(figsize=(8, 6))
county_distribution.plot(kind='bar', color='salmon')
plt.title('Top 10 Geographic Distribution by County')
plt.xlabel('County')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

