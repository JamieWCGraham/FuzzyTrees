
import numpy as np 
import pandas as pd

def generate_feature_vectors(columns_features,pair_data,distance_function):
    for jj in range(len(columns_features)):
        distances = np.zeros((len(pair_data),1))
        for ii in range(len(pair_data)):
            distance = distance_function(str(pair_data.iloc[ii,jj+1]),str(pair_data.iloc[ii,jj+6]))
            distances[ii] = (distance)
        pair_data[columns_features[jj]] = distances
    return pair_data