
from fastapi import FastAPI
from pydantic import BaseModel, create_model
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import column
import uvicorn
import json

# Création de l'API
app = FastAPI()



ohe = joblib.load('C:\\Users\\pauli\\P7\\ohe.joblib')
scaler = joblib.load ('C:\\Users\\pauli\\P7\\scaler_fit.joblib')
model = joblib.load('C:\\Users\\pauli\\P7\\model.joblib')
seuil=0.45


@app.route('/')
def home(arg):
    print(arg)
    return {'message':"Page d'accueil : Saisissez l'ID client dans l'url "}


# @app.get("/Accord") => Accord?id_client=XXXXX
@app.get("/Accord/{id_client}")
async def accord(id_client: str):
    print(id_client)
    dataframe = pd.read_csv('C:\\Users\\pauli\\P7\\test_api_client.csv', encoding ='utf-8')
    all_id_client = list(dataframe['SK_ID_CURR'].unique())

    #ID = request.args.get('id_client')
    ID = int(id_client)

    # On vérifie qu'il est bien dans la liste
    if ID not in all_id_client:
        return {"Client inconnu dans la BDD"}

    # Recupération des données
    else :

        df = dataframe[dataframe['SK_ID_CURR'] == ID]
        df['annuity_income_ratio']= df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        df['credit_annuity_ratio']= df['AMT_CREDIT'] / df['AMT_ANNUITY']
        df['credit_goods_price_ratio']= df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
        df['credit_downpayment']= df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
        df['AGE_INT']= df['DAYS_BIRTH'] / -365
        df = df.drop(['SK_ID_CURR'], axis=1)

        cat_features=df.select_dtypes(include=['object']).columns 
        num_features=df.select_dtypes(include=['int64', 'float64']).columns

        
        cat_array=ohe.transform(df[cat_features]).todense()
        num_array=df[num_features].to_numpy()

        num_array =scaler.transform(num_array)

        X=np.concatenate([cat_array, num_array], axis=1)
        X=np.asarray(X)

        probability_default_payment = model.predict_proba(X)[:, 1]
        if probability_default_payment >= seuil:
            prediction = "Prêt NON Accordé"
        else:
            prediction = "Prêt Accordé"
        return prediction