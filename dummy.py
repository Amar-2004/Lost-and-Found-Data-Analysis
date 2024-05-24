
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier


file_path = "Lost and Found.csv"
df = pd.read_csv(file_path)

df = df.drop('Roll.No', axis=1)


df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)



label_encoder = LabelEncoder()
df['Item'] = label_encoder.fit_transform(df['Item'])
df['Location'] = label_encoder.fit_transform(df['Location'])
df['F/L'] = label_encoder.fit_transform(df['F/L'])





known_locations = df[df['Location'] != 'Unknown']
unknown_locations = df[df['Location'] == 'Unknown']


X_train = known_locations.drop(['Location'], axis=1)
y_train = known_locations['Location']


X_predict = unknown_locations.drop(['Location'], axis=1)


rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)


predicted_locations = rf_model.predict(X_predict)


df.loc[df['Location'] == label_encoder.transform(['Unknown'])[0], 'Location'] = predicted_locations









df['Location'].replace('unknown', 'Unknown', inplace=True)


plt.figure(figsize=(12, 6))
sns.countplot(x='Item', data=df, palette='viridis')
plt.title('Countplot for Item Categories')
plt.xticks(rotation=45, ha='center')
plt.show()


plt.figure(figsize=(14, 8))
sns.countplot(x='Location', data=df, palette='muted')
plt.title('Lost Items Count by Location')
plt.xticks(rotation=45, ha='right')
plt.show()


plt.figure(figsize=(16, 8))
sns.lineplot(x='Date', y='Count', data=df.groupby(['Date', 'Location']).size().reset_index(name='Count'), hue='Location', marker='o')

plt.title('Lost Items Over Time by Location')
plt.xlabel('Date')
plt.ylabel('Lost Items Count')
plt.xticks(rotation=45, ha='right')
plt.show()

df_monthly = df.resample('M', on='Date').size().reset_index(name='Count')
print(df_monthly)

plt.figure(figsize=(16, 8))
sns.lineplot(x='Date', y='Count', data=df_monthly, marker='o')
plt.title('Monthly Trend of Lost Items Over Time')
plt.xlabel('Date')
plt.ylabel('Lost Items Count')
plt.xticks(rotation=45, ha='right')
plt.show()



label_encoder = LabelEncoder()
df['Item'] = label_encoder.fit_transform(df['Item'])
df['Location'] = label_encoder.fit_transform(df['Location'])
df['F/L'] = label_encoder.fit_transform(df['F/L'])


features = df[['Item', 'Location', 'F/L']]


num_clusters = 3


kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(features)


print(df[['Item', 'Location', 'F/L', 'Cluster']])




df['Cluster'] = kmeans.labels_  


item_encoder = LabelEncoder()
location_encoder = LabelEncoder()
fl_encoder = LabelEncoder()


df['Item'] = item_encoder.fit_transform(df['Item'])
df['Location'] = location_encoder.fit_transform(df['Location'])
df['F/L'] = fl_encoder.fit_transform(df['F/L'])


df['Item'] = item_encoder.inverse_transform(df['Item'])
df['Location'] = location_encoder.inverse_transform(df['Location'])
df['F/L'] = fl_encoder.inverse_transform(df['F/L'])


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Item', y='Location', hue='Cluster', data=df, palette='viridis', legend='full')
plt.title('K-Means Clustering - Item vs Location')
plt.show()





for column in df.select_dtypes(include=['datetime64']).columns:
    df[column] = df[column].astype(np.int64)
X = df.drop('Location', axis=1)
y = df['Location']


label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


logistic_regression_model = LogisticRegression(max_iter=1000)
logistic_regression_model.fit(X_train, y_train)


y_pred = logistic_regression_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred,zero_division=1)


print(f'Accuracy: {accuracy}')
print('Classification Report:\n', classification_report_str)




