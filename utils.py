import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
from scipy.stats import wishart


def train_prep(train_data, d):
    # Considérer uniquement 'Close'
    train_data = train_data['Close']


    # Scaler le dataset d'entraînement
    sc = MinMaxScaler(feature_range=(0,1))
    train_data_scaled = sc.fit_transform(train_data.values.reshape(-1, 1))

    # Créer un dataframe pandas avec des colonnes décalées dans le temps
    df_train_data = pd.DataFrame(train_data_scaled, columns=['Close'])
    for i in range(1, d+1):
        df_train_data[f'Close_{i}'] = df_train_data['Close'].shift(-i)

    # Séparer les données en entrées X et la cible y
    X_train_data = df_train_data.drop('Close', axis=1)
    y_train_data = df_train_data['Close'].shift(-(d+1))


    X_train_data.dropna(inplace=True)
    X_train_data.drop(X_train_data.index[-1], inplace=True)
    y_train_data.dropna(inplace=True)
    
    return (X_train_data, y_train_data, sc)

def predict_model_one_by_one(model, train_data, T_final, sc, d):
    # Select only the closing price
    train_data = train_data['Close']
    
    inputs = train_data[len(train_data)-d:len(train_data)]
    inputs_scaled  = sc.transform(inputs.values.reshape(-1, 1))
    #print(inputs_scaled)

    # Créer un dataframe pandas avec des colonnes décalées dans le temps
    df_train_data = pd.DataFrame(inputs_scaled, columns=['Close'])
    for i in range(1, d+1):
        df_train_data[f'Close_{i}'] = df_train_data['Close'].shift(-i+1)

    # Itérer successivement sur les données pour prédire
    predicted_stock_price = []
    X_test_data = df_train_data.drop('Close', axis=1).iloc[0:2].copy()
    for i in range(T_final):
        #print(f"-------------------------{i}-------------------------")
        #print("X_test_data=", X_test_data)
        X_test_data.iloc[1][f"Close_{d}"] = 0 # Pour retirer le NaN (la deuxième ligne ne nous intéresse pas)
        
        # Prédiction
        pred = model.predict(X_test_data)
        #print(f"pred_{i}=", pred)
        predicted_stock_price.append(pred[0])
        
        # Ajout de la prédiction pour le prochain test
        X_test_data.iloc[1][f"Close_{d}"] = pred[0]
        
        new_line = X_test_data.iloc[1].shift(-1)
        #print("new_line=", new_line)
        X_test_data = pd.concat([X_test_data, new_line.to_frame().T], ignore_index=True, axis=0)
        # Supprimer la première ligne
        X_test_data.drop(X_test_data.index[0], inplace=True)

    predicted_stock_price = sc.inverse_transform(np.array(predicted_stock_price).reshape(-1, 1))
    
    return predicted_stock_price

def generate_covariance_matrix(d, method="wishart"):
    if method == "wishart":
        # Génération des matrices de covariances pour les X selon une Wishart(I, d-1, d-1)
        C_s = wishart.rvs(df=d-1, scale=np.eye(d-1))
        C_b = wishart.rvs(df=d-1, scale=np.eye(d-1))
    
    elif method == "spiked":
        # Génération des matrices de covariances pour les X selon le modèle de covariance spiked
        u = np.random.uniform(low=-1, high=1, size=d-1)
        v = np.random.uniform(low=-1, high=1, size=d-1)
        rho = d
        C_s = np.eye(d-1) + rho*np.dot(np.transpose(u), u)
        C_b = np.eye(d-1) + rho*np.dot(np.transpose(v), v)
        
    else:
        print("Please choose a method between 'spiked' and 'wishart'.")
    return C_s, C_b

def plot_predictions(test, predicted):
    plt.plot(test, color='red',label='Cours vrai')
    plt.plot(predicted, color='blue',label='Cours prédit')
    plt.xlabel('Temps')
    plt.ylabel('Cours de fermeture (USD)')
    plt.legend()
    plt.show()

def return_rmse(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("RMSE = {}.".format(rmse))
    return rmse

def return_relative_bias(test, predicted):
    return np.sum(predicted - test) / np.sum(test)

def return_rb_gamma(n, gamma):
    return math.sqrt(n)*np.linalg.norm(gamma, ord=1)

