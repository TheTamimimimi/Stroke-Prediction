import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest
from scipy.stats.mstats import winsorize
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import joblib


dataset = pd.read_csv(r"C:\Users\7mood\Desktop\2023-2\Data Mining\stroke-prediction-dataset-assignment\train.csv")
testset = pd.read_csv(r"C:\Users\7mood\Desktop\2023-2\Data Mining\stroke-prediction-dataset-assignment\test.csv")


#Change the categorical values to dummy values
le = LabelEncoder()
cols=dataset.select_dtypes(include=['object']).columns
dataset[cols]=dataset[cols].apply(le.fit_transform)
testset[cols]=testset[cols].apply(le.fit_transform)


# remove null values
dataset = dataset.replace([np.inf, -np.inf], np.nan)
dataset = dataset.fillna(dataset.mean())

# Select Features
featureClassifier = SelectKBest(k='all')
fits = featureClassifier.fit(dataset.drop('stroke',axis=1),dataset['stroke'])
x=pd.DataFrame(fits.scores_)
columns = pd.DataFrame(dataset.drop('stroke',axis=1).columns)
fscores = pd.concat([columns,x],axis=1)
fscores.columns = ['Attribute','Score']
fscores.sort_values(by='Score',ascending=False)


# print best featrues
cols=fscores[fscores['Score']>40]['Attribute']
print(cols)


#The columns that is needed only.
traincols = ['age','hypertension','heart_disease','ever_married','avg_glucose_level']
testcols =['id','age','hypertension','heart_disease','ever_married','avg_glucose_level']
testset = testset[testcols]


# Prepare data for stroke prediction
x = dataset[traincols]
y = dataset['stroke']


x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=42)

# SMOTE to balance the data
smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)

classifier = SVC(random_state=42, kernel='rbf', C= 1)




# Cluster the data
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42,init='k-means++')
train_clusters = kmeans.fit_predict(x_train)


print(type(x_train))
# Train a separate model for each cluster
for i in range(num_clusters):
    cluster_idx = np.where(train_clusters == i)

    x_train_cluster = x_train.iloc[cluster_idx]
    y_train_cluster = y_train.iloc[cluster_idx]

    pipeline = Pipeline([('classifier', classifier)])
    pipeline.fit(x_train_cluster, y_train_cluster.ravel())
    
    # Save the trained model for each cluster
    joblib.dump(pipeline, f'model_cluster_{i}.pkl')

# Predictions
valid_clusters = kmeans.predict(x_valid)
y_pred = np.empty_like(y_valid)

for i in range(num_clusters):
    cluster_idx = np.where(valid_clusters == i)
    
    # Load the trained model for each cluster
    pipeline = joblib.load(f'model_cluster_{i}.pkl')
    
    x_valid_cluster = x_valid.iloc[cluster_idx]
    y_pred_cluster = pipeline.predict(x_valid_cluster)
    
    y_pred[cluster_idx] = y_pred_cluster

# Calculate the accuracy
accuracy = accuracy_score(y_valid, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Print the classification report
print(classification_report(y_valid, y_pred))

# Print the confusion matrix
print(confusion_matrix(y_valid, y_pred))

test_ids = testset['id']

# Drop the 'id' column, as it is not useful for prediction
testset= testset.drop('id', axis=1)

# Assign clusters to the test dataset
test_clusters = kmeans.predict(testset)

# Create an empty array to store predictions
test_predictions = np.empty(testset.shape[0], dtype=int)

# Make predictions for each cluster
for i in range(num_clusters):
    # Get instances of the testset that were assigned to the current cluster
    cluster_idx = np.where(test_clusters == i)
    testset_cluster = testset.iloc[cluster_idx]
    
    # Load the pipeline trained on the current cluster
    pipeline = joblib.load(f'model_cluster_{i}.pkl')
    
    # Make predictions on the current cluster
    test_predictions_cluster = pipeline.predict(testset_cluster)
    
    # Store predictions in the predictions array
    test_predictions[cluster_idx] = test_predictions_cluster

# Combine the 'id' column with the predictions
submission = pd.DataFrame({'id': test_ids, 'stroke': test_predictions})

# Save the submission file to a CSV
submission.to_csv('stroke_predictions_submission.csv', index=False)

