
from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
import json
from fastapi.encoders import jsonable_encoder

# Création de l'API
app = FastAPI()



model = joblib.load('./clXGB.joblib')



# @app.route('/')
# def home(arg):
#     print(arg)
#     return {'message':"Page d'accueil : Saisissez l'ID client dans l'url "}


# @app.get("/Accord") => Accord?id_client=XXXXX
@app.get("/Accord/{id_client}")
async def accord(id_client: str):
    print(id_client)
    #dataframe = pd.read_csv('./bin/test_api_client.csv', encoding ='utf-8')
    df_train = pd.read_csv('./df_train.csv', encoding ='utf-8')
    df_train2 = pd.read_csv('./df_train2.csv', encoding ='utf-8')
    all_id_client = list(df_train['SK_ID_CURR'].unique())

    #ID = request.args.get('id_client')
    ID = int(id_client)

    # On vérifie qu'il est bien dans la liste
    if ID not in all_id_client:
        return {"Client inconnu dans la BDD"}

    # Recupération des données
    else :

        ligne_client = df_train2[df_train['SK_ID_CURR'] == ID]
        # df['annuity_income_ratio']= df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        # df['credit_annuity_ratio']= df['AMT_CREDIT'] / df['AMT_ANNUITY']
        # df['credit_goods_price_ratio']= df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
        # df['credit_downpayment']= df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
        # df['AGE_INT']= df['DAYS_BIRTH'] / -365
        # df = df.drop(['SK_ID_CURR'], axis=1)

        # cat_features=df.select_dtypes(include=['object']).columns 
        # num_features=df.select_dtypes(include=['int64', 'float64']).columns

        
        # cat_array=ohe.transform(df[cat_features]).todense()
        # num_array=df[num_features].to_numpy()

        # num_array =scaler.transform(num_array)

        # X=np.concatenate([cat_array, num_array], axis=1)
        # X=np.asarray(X)

        
        #probability_default_payment = model.predict_proba(X)[:, 1]
        #return json.dumps(model.predict_proba(X).tolist())

        #return json.dumps(model.predict_proba(ligne_client).tolist())

        # if probability_default_payment >= seuil:
        #     prediction = "Prêt NON Accordé"
        # else:
        #     prediction = "Prêt Accordé"
        # return prediction
        proba = model.predict_proba(ligne_client)
        #pred = int(model.predict(ligne_client)[0])
        proba_0 = float(proba[0][0])
        #proba_1 = str(round(proba[0][1]*100,1)) + '%'
        result = {'proba_0': proba_0}
        return jsonable_encoder(result)
