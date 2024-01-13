import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


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

def predict_model_one_by_one(model, f_predict, train_data, T_final, sc, d):
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
        pred = f_predict(model, X_test_data)
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