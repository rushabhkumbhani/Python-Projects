import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle


def data_split(data, ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data) * ratio)
    test_indices=shuffled[:test_set_size] # All rows and test set size columns
    train_indices=shuffled[test_set_size:] # Basic Slicing 
    return data.iloc[train_indices], data.iloc[test_indices]




if __name__=="__main__":

    # Read the data

    df=pd.read_csv("C://Users//DELL//Desktop//Vs Code Python//Corona Virus Solution//")
    train, test = data_split(df, 0.2)

    x_train=train[['fever','bodyPain','age','runnyNose','diffBreath']].to_numpy()
    x_test=test[['fever','bodyPain','age','runnyNose','diffBreath']].to_numpy()

    y_train=train[['infectionProb']].to_numpy().reshape(2060,)
    y_test=test[['infectionProb']].to_numpy().reshape(515,)

    clf=LogisticRegression()
    clf.fit(x_train, y_train)

    # Open a file where you want to store the data

    file=open('model.pkl','wb')

    # Dump information to that file

    pickle.dump(clf, file)
    file.close()

    # Code for Inference

    inputFeatures = [102,1,22,-1,1]
    infProb = clf.predict_proba([inputFeatures])[0][1]