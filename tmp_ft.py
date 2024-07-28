# if train_random_forest:
import jellyfish as j
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
import fuzzywuzzy as f
from fuzzywuzzy import fuzz
from joblib import dump, load

train_random_forest = True


if train_random_forest:
    og_pairs = pd.read_csv("training_data_zips.csv")
    pair_data = og_pairs
    pair_data.head()

# convert all strings to lower case and remove spaces and commas. 
if train_random_forest:
    pair_data.loc[:, pair_data.dtypes == 'object'] = pair_data.select_dtypes(['object']).apply(lambda x: x.str.lower())
    pair_data.loc[:, pair_data.dtypes == 'object'] = pair_data.select_dtypes(['object']).apply(lambda x: x.str.replace('[ ,-]', '', regex=True))
    pair_data.loc[:, pair_data.dtypes == 'object'] = pair_data.select_dtypes(['object']).apply(lambda x: x.str.replace('[&]', 'and', regex=True))
    pair_data['og_idx'] = pair_data.index
    pair_data = pair_data.sample(frac=1).reset_index(drop=True)
    pair_data.head()
# pair_data = pair_data.iloc[1:10,:]


def jaro_winkler_distance(str1,str2):
    score = j.jaro_winkler_similarity(str1,str2)
    return score

def levenshtein_distance(str1,str2):
    score = f.fuzz.ratio(str1, str2)/100
    return score    


def generate_feature_vectors(columns_features,pair_data,distance_function):
    for jj in range(len(columns_features)):
        print("Creating feature vector for: " + columns_features[jj])
        distances = np.zeros((len(pair_data),1))
        for ii in range(len(pair_data)):
            distance = distance_function(str(pair_data.iloc[ii,jj+1]),str(pair_data.iloc[ii,jj+6]))
            distances[ii] = (distance)
        pair_data[columns_features[jj]] = distances
    return pair_data



if train_random_forest:

    lev_features = ['name_lev_dist','city_lev_dist','address_lev_dist','zip_lev_dist'] 
    jaro_features = ['name_jaro_dist','city_jaro_dist','address_jaro_dist','zip_jaro_dist']
    pair_data = generate_feature_vectors(lev_features,pair_data,levenshtein_distance)
    pair_data = generate_feature_vectors(jaro_features,pair_data,jaro_winkler_distance)
    pair_data['TOTAL_SCORE'] = pair_data.iloc[:,-8:].sum(axis=1)


lev_features = ['name_lev_dist','city_lev_dist','address_lev_dist','zip_lev_dist'] 
jaro_features = ['name_jaro_dist','city_jaro_dist','address_jaro_dist','zip_jaro_dist']


example = pd.read_csv("example.csv")

example = generate_feature_vectors(lev_features,example,levenshtein_distance)
example = generate_feature_vectors(jaro_features,example,jaro_winkler_distance)
example['TOTAL_SCORE'] = example.iloc[:,-8:].sum(axis=1)


if train_random_forest:
    columns_features = ['og_idx','name_lev_dist','city_lev_dist','address_lev_dist','zip_lev_dist','name_jaro_dist','city_jaro_dist','address_jaro_dist','zip_jaro_dist','TOTAL_SCORE']
    feature_df = pair_data.loc[:,columns_features]
    target_df = pair_data.loc[:,'is_match']

    X = np.array(feature_df.iloc[:,1:])  # Features
    y = np.array(target_df)  # Target variable

    # First, split the data into training + validation set and test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Now, split the training + validation set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=1)
    
    # Initialize the Random Forest classifier
    random_forest_classifier = RandomForestClassifier(n_estimators=200, random_state=1)

    # Fit the model on the training data
    random_forest_classifier.fit(X_train, y_train)

    # Predict the labels of the test set
    y_pred = random_forest_classifier.predict(X_val)
    y_pred_test = random_forest_classifier.predict(X_test)

    # # Calculate the accuracy of the model
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Model Accuracy: {accuracy*100:.5f}%")

if train_random_forest:
    # Accuracy
    accuracy = accuracy_score(y_val, y_pred)
    print(f'Validation Set Accuracy: {accuracy:.5f}')

    # Precision
    precision = precision_score(y_val, y_pred)
    print(f'Validation Set Precision: {precision:.5f}')

    # Recall
    recall = recall_score(y_val, y_pred)
    print(f'Validation Set Recall: {recall:.5f}')

    # F1 Score
    f1 = f1_score(y_val, y_pred)
    print(f'Validation Set F1 Score: {f1:.5f}')

    # ROC-AUC
    roc_auc = roc_auc_score(y_val, y_pred)
    print(f'Validation Set ROC-AUC: {roc_auc:.5f}')

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    print('Validation Set Confusion Matrix:')
    print(cm)

    # Optionally, plot ROC curve
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_val, y_pred)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()

    # Plot ROC curve
    fig.add_trace(go.Scatter(x=fpr, y=tpr,
                        mode='lines',
                        name=f'ROC curve (area = {roc_auc:.5f})',
                        line=dict(color='darkorange', width=2)))

    # Plot diagonal line
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Random Guessing',
                        line=dict(color='navy', width=2, dash='dash')))

    # Set layout
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        title='Validation Set ROC',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        legend=dict(x=0.02, y=0.98),
    )

    fig.show()
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred_test)
    print(f'Test Set Accuracy: {accuracy:.5f}')

    # Precision
    precision = precision_score(y_test, y_pred_test)
    print(f'Test Set Precision: {precision:.5f}')

    # Recall
    recall = recall_score(y_test, y_pred_test)
    print(f'Test Set Recall: {recall:.5f}')

    # F1 Score
    f1 = f1_score(y_test, y_pred_test)
    print(f'Test Set F1 Score: {f1:.5f}')

    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_test)
    print(f'Test Set ROC-AUC: {roc_auc:.5f}')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)
    print('Test Set Confusion Matrix:')
    print(cm)

    fpr, tpr, _ = roc_curve(y_test, y_pred_test)
    roc_auc = auc(fpr, tpr)


    fig = go.Figure()

    # Plot ROC curve
    fig.add_trace(go.Scatter(x=fpr, y=tpr,
                        mode='lines',
                        name=f'ROC curve (area = {roc_auc:.5f})',
                        line=dict(color='darkorange', width=2)))

    # Plot diagonal line
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Random Guessing',
                        line=dict(color='navy', width=2, dash='dash')))

    # Set layout
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        title='Test Set ROC',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        legend=dict(x=0.02, y=0.98),
    )

    fig.show()
