

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pickle

pd.set_option('display.max_columns', None)
df = pd.read_csv("Dataset/data.csv")
df = df.dropna(axis=1, how='all') # dataset has extra empty column

df.info()

print("describe: \n", df.describe(), "\n")

print("Number of duplicated values:", df.duplicated().sum(), "\n")

print("Number of null values: \n", df.isnull().sum(), "\n")

df2 = df.copy() # dataset with dignosis M and B converted to 1 and 0
df2['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

df3 = df2.drop(columns =['diagnosis', 'id']) # diagnosis and id column removed 
diagnosis_outcome = df2.diagnosis

plt.figure()
sns.countplot(x=df2['diagnosis'],  palette='Set1')



sns.heatmap(df3.corr(), annot=True)

# drop columns with over 81% correlation, 14 dropped, 16 remaining
df4 = df3.drop(columns = [
    'perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean', 
    'perimeter_se', 'area_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 
    'area_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'fractal_dimension_worst'
    ])
plt.figure()
plt.show()

# scales data between 0 and 1
scaler = MinMaxScaler()
scaler.fit(df4)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
    
rescaled_df = pd.DataFrame(scaler.transform(df4), columns=df4.columns) # using 16 features

    