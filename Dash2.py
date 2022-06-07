from ast import If
import pickle
from pyexpat import model
from urllib.request import proxy_bypass
import streamlit as st
import joblib
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
#from sklearn import model_selection
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
#import dvc.api
#import lightgbm
from matplotlib.image import imread
import altair as alt
import requests

#from vega_datasets import data

#from FastAPI.Dashboard import comparaison, get_client, infos_client, score_viz

seuil=0.45

def load_data():

    # Chargement du modèle pré-entrainé
    model = pickle.load(open('../xgb_model.pkl', 'rb'))

    # Chargement du modèle pré-entrainé	=> Appel API
    df_train = pd.read_csv('./df_train.csv', encoding='utf-8', index_col=False)
    df_train2 = pd.read_csv('./df_train2.csv', encoding='utf-8', index_col=False)

    # df_train=df_train.reset_index(drop=True)

    # Recupération des données pour modele
    ohe = joblib.load('./bin/ohe.joblib')
    scaler = joblib.load('./bin/scaler_fit.joblib')
    #model = joblib.load('./FastAPI/bin/model.joblib')
    #clXGB
    model = joblib.load('../clXGB.joblib')

    df_train['YEARS_BIRTH']=(df_train['DAYS_BIRTH']/-365).apply(lambda x: int(x))
    # df = pd.read_csv('./bin/test_api_client.csv', encoding='utf-8')
    # df.dropna()
    # df['annuity_income_ratio'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    # df['credit_annuity_ratio'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    # df['credit_goods_price_ratio'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    # df['credit_downpayment'] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
    # df['AGE_INT'] = df['DAYS_BIRTH'] / -365
    # df = df.drop(['SK_ID_CURR'], axis=1)

    # Encodage
    # cat_features = df.select_dtypes(include=['object']).columns
    # num_features = df.select_dtypes(include=['int64', 'float64']).columns

    # cat_array = ohe.transform(df[cat_features]).todense()
    # num_array = df[num_features].to_numpy()

    # num_array = scaler.transform(num_array)

    # X = np.concatenate([cat_array, num_array], axis=1)
    # df_train2 = pd.DataFrame(X)
    # #df_train2 = pd.get_dummies(df_train2)
    # df_train2 = df_train2.fillna(df_train2.median())
    # X = np.asarray(X)

    # Import du fichier encodé et standardisé
    #df_train2  = pd.read_csv('../df_train2.csv', encoding ='utf-8')

    # Chargement des données de test après encodage
    logo = imread("./../logo.png")

    # Calcul des SHAP values
    explainer = shap.TreeExplainer(model)
    #shap_values = explainer.shap_values(X)
    #shap_values = explainer.shap_values(df_train2)[1]
    df_X_test=pd.DataFrame(data=df_train2,columns=df_train2.columns)
    shap_values = explainer.shap_values(df_X_test)
    exp_value = explainer.expected_value
    return df_train, df_train2, shap_values, model, exp_value, logo
    # return df_train,df_train2,model,logo


def tab_client(df_train):
    '''Fonction pour afficher le tableau du portefeuille client avec un système de 6 champs de filtres
    permettant une recherche plus précise.
    Le paramètre est le dataframe des clients
    '''
    #st.title('Dashboard Pret à dépenser')
    st.subheader('Tableau des clients')
    row0_1, row0_spacer2, row0_2, row0_spacer3, row0_3, row0_spacer4, row_spacer5 = st.columns([
                                                                                               1, .1, 1, .1, 1, .1, 4])

    # # #Définition des filtres via selectbox
    sex = row0_1.selectbox(
        "Sexe", ['All']+df_train['CODE_GENDER'].unique().tolist())
    age = row0_1.selectbox(
        "Age", ['All']+(np.sort(df_train['YEARS_BIRTH'].unique()).astype(str).tolist()))
    fam = row0_2.selectbox("Statut familial", [
                           'All']+df_train['NAME_FAMILY_STATUS'].unique().tolist())
    child = row0_2.selectbox("Enfants", [
                             'All']+(np.sort(df_train['CNT_CHILDREN'].unique()).astype(str).tolist()))
    pro = row0_3.selectbox(
        "Statut pro.", ['All']+df_train['NAME_INCOME_TYPE'].unique().tolist())
    stud = row0_3.selectbox(
        "Niveau d'études", ['All']+df_train['NAME_EDUCATION_TYPE'].unique().tolist())

    # Affichage du dataframe selon les filtres définis
    db_display = df_train[['SK_ID_CURR', 'CODE_GENDER', 'YEARS_BIRTH', 'NAME_FAMILY_STATUS', 'CNT_CHILDREN',
                          'NAME_EDUCATION_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_HOUSING_TYPE',
                          'NAME_INCOME_TYPE', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']]
    db_display['YEARS_BIRTH'] = db_display['YEARS_BIRTH'].astype(str)
    db_display['CNT_CHILDREN'] = db_display['CNT_CHILDREN'].astype(str)
    db_display['AMT_INCOME_TOTAL'] = df_train['AMT_INCOME_TOTAL'].apply(
        lambda x: int(x))
    db_display['AMT_CREDIT'] = df_train['AMT_CREDIT'].apply(lambda x: int(x))
    db_display['AMT_ANNUITY'] = df_train['AMT_ANNUITY'].apply(
        lambda x: x if pd.isna(x) else int(x))

    db_display = filter(db_display, 'CODE_GENDER', sex)
    db_display = filter(db_display, 'YEARS_BIRTH', age)
    db_display = filter(db_display, 'NAME_FAMILY_STATUS', fam)
    db_display = filter(db_display, 'CNT_CHILDREN', child)
    db_display = filter(db_display, 'NAME_INCOME_TYPE', pro)
    db_display = filter(db_display, 'NAME_EDUCATION_TYPE', stud)
    # st.dataframe(db_display)

    st.dataframe(df_train)
    # st.dataframe(db_display)

    st.markdown("**Total clients correspondants: **"+str(len(db_display)))

    # Affichage en dessous des histo de répartitions
    source = db_display

    base = alt.Chart(source)

    bar = base.mark_bar().encode(
        x=alt.X('CNT_CHILDREN:Q', bin=True, axis=None),
        y='count()'
    )

    rule = base.mark_rule(color='red').encode(
        x='mean(CNT_CHILDREN):Q',
        size=alt.value(3)
    )

    bar + rule


def histo_client(df):
    '''Fonction pour afficher le tableau du portefeuille client avec un système de 6 champs de filtres
    permettant une recherche plus précise.
    Le paramètre est le dataframe des clients
    '''
    #st.title('Dashboard Pret à dépenser')
    st.subheader('Vue ensemble des clients')

    # Affichage en dessous des histo de répartitions
    source = df

    base = alt.Chart(source)

    bar = base.mark_bar().encode(
        x=alt.X('CNT_CHILDREN:Q', bin=True, axis=None),
        y='count()'
    )

    rule = base.mark_rule(color='red').encode(
        x='mean(CNT_CHILDREN):Q',
        size=alt.value(3)
    )

    bar + rule


def filter(df, col, value):
    '''Fonction pour filtrer le dataframe selon la colonne et la valeur définies'''
    if value != 'All':
        db_filtered = df.loc[df[col] == value]
    else:
        db_filtered = df
    return db_filtered


def score_viz(model, df_train2, client, idx_client, exp_value, shap_values):
    """Fonction principale de l'onglet 'Score visualisation' """
    #st.title('Dashboard Pret à dépenser')
    st.subheader('Visualisation score')

    score, result = prediction(model, df_train2, client)
    st.subheader(result)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number={'font': {'size': 48}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': result.tolist(), 'font': {'size': 28,
                                                 'color': color(result)}},
        delta={'reference': 0.48, 'increasing': {
            'color': "red"}, 'decreasing': {'color': 'green'}},
        gauge={
            'axis': {'range': [0, 1], 'tickcolor': color(result)},
            'bar': {'color': color(result)},
            'steps': [
                {'range': [0, 0.48], 'color': 'lightgreen'},
                {'range': [0.48, 1], 'color': 'lightcoral'}],
            'threshold': {
                'line': {'color': "black", 'width': 5},
                'thickness': 1,
                'value': 0.48}}))

    st.plotly_chart(fig)

    st_shap(shap.force_plot(
        exp_value, shap_values[idx_client], features=df_train2.iloc[idx_client], feature_names=df_train2.columns, figsize=(12, 5)))


def prediction(model, df_train2, id):
    '''Fonction permettant de prédire la capacité du client à rembourser son emprunt.
    les paramètres sont le modèle, le dataframe et l'ID du client'''
    
    # api-endpoint
    URL = "https://fast-api-scoringbancaire.herokuapp.com/Accord/" + str(id)
  
    # location given here
    location = "delhi technological university"
  
    # defining a params dict for the parameters to be sent to the API
    #PARAMS = {'Accord':id}
  
    # sending get request and saving the response as response object
    r = requests.get(url = URL)#, params = PARAMS)
  
    # extracting data in json format
    #data = r.json()
    
    #y_pred = model.predict_proba(df_train2)[id, 1]
    result=r.json()
    proba_1 = result['proba_1']
    y_pred = float(proba_1)
    decision = np.where(y_pred > seuil, "Rejected", "Approved")
    
    return y_pred, decision


def color(pred):
    '''Définition de la couleur selon la prédiction'''
    if pred == 'Approved':
        col = 'Green'
    else:
        col = 'Red'
    return col

def get_client(df_train):
    #"""Sélection d'un client via une selectbox"""
    client = st.sidebar.selectbox('Client', df_train['SK_ID_CURR'])
    idx_client = df_train.index[df_train['SK_ID_CURR'] == client][0]
    return client, idx_client

def Importance_feature(df_train2, model):
    #st_shap(shap.force_plot( exp_value, shap_values[client], features=df_train2.iloc[client], feature_names=df_train2.columns, figsize=(12, 5)))
    #st_shap(shap.plots.bar(shap_values, max_display=12))
    #st_shap(shap.plots.waterfall(shap_values[client]), height=300)
    #st_shap(shap.plots.beeswarm(shap_values), height=300)

    st.subheader('Feature importance')
    #fig, ax = plt.subplots()
    feat_importance=model.feature_importances_
    data = pd.DataFrame(data = {'Names' : df_train2.columns, 'Importance':feat_importance}).sort_values(by='Importance')
    data['Importance']=data['Importance']/data['Importance'].max()
    # Affichage des 20 features les plus importantes 
    fig=plt.figure(figsize=(16,12))
    plt.xlabel("Importance").set_fontsize(20)
    plt.ylabel("Feature").set_fontsize(20)
    plt.tick_params(axis = 'both', labelsize = 20)
    #plt.title("Feature importance")
    plt.barh(data['Names'].tail(15) ,data['Importance'].tail(15))
 
    plt.legend(fontsize=12)
    st.pyplot(fig)


def st_shap(plot, height=None):
    """Fonction permettant l'affichage de graphique shap values"""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)



def infos_client(df_train, client, idx_client):
    """Affichage des infos du client sélectionné dans la barre latérale"""
    st.sidebar.markdown("**ID client: **"+str(client))
    st.sidebar.markdown("**Sexe: **"+df_train.loc[idx_client, 'CODE_GENDER'])
    st.sidebar.markdown("**Statut familial: **" +
                        df_train.loc[idx_client, 'NAME_FAMILY_STATUS'])
    st.sidebar.markdown(
        "**Enfants: **"+str(df_train.loc[idx_client, 'CNT_CHILDREN']))
    st.sidebar.markdown(
        "**Age: **"+str(df_train.loc[idx_client, 'YEARS_BIRTH']))
    st.sidebar.markdown("**Statut pro.: **" +
                        df_train.loc[idx_client, 'NAME_INCOME_TYPE'])
    st.sidebar.markdown("**Niveau d'études: **" +
                        df_train.loc[idx_client, 'NAME_EDUCATION_TYPE'])


def comparaison(df_train2, df_train, idx_client):
    """Fonction principale de l'onglet 'Comparaison clientèle' """
    #st.title('Dashboard Pret à dépenser')
    st.subheader('Comparaison clientèle')
    idx_neigh, total = get_neigh(df_train2, idx_client)
    db_neigh = df_train.loc[idx_neigh, ['SK_ID_CURR', 'CODE_GENDER', 'YEARS_BIRTH', 'NAME_FAMILY_STATUS', 'CNT_CHILDREN',
                                       'NAME_EDUCATION_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_HOUSING_TYPE',
                                       'NAME_INCOME_TYPE', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'TARGET']]
    db_neigh['AMT_INCOME_TOTAL'] = db_neigh['AMT_INCOME_TOTAL'].apply(
        lambda x: int(x))
    db_neigh['AMT_CREDIT'] = db_neigh['AMT_CREDIT'].apply(lambda x: int(x))
    db_neigh['AMT_ANNUITY'] = db_neigh['AMT_ANNUITY'].apply(
        lambda x: x if pd.isna(x) else int(x))

    if total:
        display_charts(df_train, idx_client)

    else:
        display_charts(db_neigh, idx_client)


def get_neigh(df, idx_client):
    """Calcul des voisins les plus proches du client sélectionné
    Sélection du nombre de voisins par un slider.
    Retourne les proches voisins et un booléen indiquant la clientèle globale ou non"""
    row1, row_spacer1, row2, row_spacer2 = st.columns([1, .1, .3, 3])
    size = row1.slider("Taille du groupe de comparaison",
                       min_value=10, max_value=1000, value=500)
    row2.write('')
    total = row2.button(label="Clientèle globale")
    neigh = NearestNeighbors(n_neighbors=size)
    neigh.fit(df)
    k_neigh = neigh.kneighbors(
        df.loc[idx_client].values.reshape(1, -1), return_distance=False)[0]
    k_neigh = np.sort(k_neigh)
    return k_neigh, total


def display_charts(df, client):
    """Affichae des graphes de comparaison pour le client sélectionné """
    row1_1, row1_2, row1_3 = st.columns(3)
    st.write('')
    row2_10, row2_2, row2_3 = st.columns(3)

    chart_kde("Répartition de l'age", row1_1, df, 'YEARS_BIRTH', client)
    chart_kde("Répartition des revenus", row1_2,
              df, 'AMT_INCOME_TOTAL', client)
    chart_bar("Répartition du nombre d'enfants",
              row1_3, df, 'CNT_CHILDREN', client)

    chart_bar("Répartition du statut professionel",
              row2_10, df, 'NAME_INCOME_TYPE', client)
    chart_bar("Répartition du niveau d'études", row2_2,
              df, 'NAME_EDUCATION_TYPE', client)
    chart_bar("Répartition du type de logement",
              row2_3, df, 'NAME_HOUSING_TYPE', client)
    st.dataframe(df[['SK_ID_CURR', 'CODE_GENDER', 'YEARS_BIRTH', 'NAME_FAMILY_STATUS', 'CNT_CHILDREN',
                     'NAME_EDUCATION_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_HOUSING_TYPE',
                     'NAME_INCOME_TYPE', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']])


def chart_kde(title, row, df, col, client):
    """Définition des graphes KDE avec une ligne verticale indiquant la position du client"""
    with row:
        st.subheader(title)
        fig, ax = plt.subplots()
        sns.kdeplot(df.loc[df['TARGET'] == 0, col],
                    color='green', label='Target == 0')
        sns.kdeplot(df.loc[df['TARGET'] == 1, col],
                    color='red', label='Target == 1')
        plt.axvline(x=df.loc[client, col], ymax=0.95, color='black')
        plt.legend()
        st.pyplot(fig)


def chart_bar(title, row, df, col, client):
    """Définition des graphes barres avec une ligne horizontale indiquant la position du client"""
    with row:
        st.subheader(title)
        fig, ax = plt.subplots()
        data = df[['TARGET', col]]
        if data[col].dtypes != 'object':
            data[col] = data[col].astype('str')

            data1 = round(data[col].loc[data['TARGET'] == 1].value_counts(
            )/data[col].loc[data['TARGET'] == 1].value_counts().sum()*100, 2)
            data0 = round(data[col].loc[data['TARGET'] == 0].value_counts(
            )/data[col].loc[data['TARGET'] == 0].value_counts().sum()*100, 2)
            data = pd.concat([pd.DataFrame({"Pourcentage": data0, 'TARGET': 0}), pd.DataFrame(
                {'Pourcentage': data1, 'TARGET': 1})]).reset_index().rename(columns={'index': col})
            sns.barplot(data=data, x='Pourcentage', y=col, hue='TARGET', palette=[
                        'green', 'red'], order=sorted(data[col].unique()))

            data[col] = data[col].astype('int64')

            plt.axhline(y=sorted(data[col].unique()).index(
                df.loc[client, col]), xmax=0.95, color='black', linewidth=4)
            st.pyplot(fig)
        else:

            data1 = round(data[col].loc[data['TARGET'] == 1].value_counts(
            )/data[col].loc[data['TARGET'] == 1].value_counts().sum()*100, 2)
            data0 = round(data[col].loc[data['TARGET'] == 0].value_counts(
            )/data[col].loc[data['TARGET'] == 0].value_counts().sum()*100, 2)
            data = pd.concat([pd.DataFrame({"Pourcentage": data0, 'TARGET': 0}), pd.DataFrame(
                {'Pourcentage': data1, 'TARGET': 1})]).reset_index().rename(columns={'index': col})
            sns.barplot(data=data, x='Pourcentage', y=col, hue='TARGET', palette=[
                        'green', 'red'], order=sorted(data[col].unique()))

            plt.axhline(y=sorted(data[col].unique()).index(
                df.loc[client, col]), xmax=0.95, color='black', linewidth=4)
            st.pyplot(fig)


def main():
    df_train, df_train2, shap_values, model, exp_value, logo = load_data()
    
    st.sidebar.image(logo)

    PAGES = ["Score du client", "Ensemble des clients", "Tableau des clients", "Comparaison"]

    st.sidebar.write('')
    st.sidebar.write('')

    st.sidebar.title('Dashboard Pret à dépenser')
    selection = st.sidebar.radio("Menu", PAGES)

    if selection == "Tableau des clients":
        tab_client(df_train)
        # if selection=="Ensemble":
        #  	histo_client(df)
    if selection == "Score du client":
        client, idx_client = get_client(df_train)
        infos_client(df_train, client, idx_client)
        score_viz(model, df_train2, client, idx_client, exp_value, shap_values)
    if selection == "Comparaison":
        client, idx_client = get_client(df_train)
        infos_client(df_train, client, idx_client)
        comparaison(df_train2, df_train, idx_client)
    if selection == "Ensemble des clients":
        #histo_client(df_train)
        #client, idx_client = get_client(df_train)
        #infos_client(df_train, client, idx_client)
        #Importance_feature(shap_values, client)
        Importance_feature(df_train2, model)


if __name__ == '__main__':
    main()
