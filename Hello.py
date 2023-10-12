# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import requests
import streamlit as st
from streamlit.logger import get_logger
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import plotly.figure_factory as ff
import scipy.stats as stats
import matplotlib.pyplot as plt
from streamlit_folium import folium_static
import folium

LOGGER = get_logger(__name__)

def LaadDataAPI(URLAPI):
    ###Inladen API - kijk naar country code en maxresults
    response = requests.get(URLAPI)
    ###Omzetten naar dictionary
    responsejson  = response.json()
    response.json()
    #Met json_normalize zet je de eerste kolom om naar losse kolommen
    df = pd.json_normalize(responsejson)
    return df

def HistLaadpalen(data):
  data  = data[(data['ChargeTime'] > 0) & (data['ChargeTime'] < 10)]
  laadtijd = data['ChargeTime']

  # Histogram van laadtijd
  plt.hist(laadtijd, bins=20, density=True, color='blue', edgecolor='black')
  plt.title('Histogram van Laadtijd')
  plt.xlabel('Laadtijd (seconden)')
  plt.ylabel('Kansdichtheid')


  # Bereken het gemiddelde en de mediaan
  gemiddelde = np.mean(laadtijd)
  mediaan = np.median(laadtijd)


  # Voeg annotaties toe voor gemiddelde en mediaan
  plt.axvline(gemiddelde, color='red', linestyle='dashed', linewidth=2, label=f'Gemiddelde: {gemiddelde:.2f}')
  plt.axvline(mediaan, color='green', linestyle='dashed', linewidth=2, label=f'Mediaan: {mediaan}')

  # Benadering van de kansdichtheidsfunctie (PDF)
  x = np.linspace(min(laadtijd), max(laadtijd), 100)
  pdf = stats.norm.pdf(x, loc=gemiddelde, scale=np.std(laadtijd))
  plt.plot(x, pdf, color='orange', label='Kansdichtheidsfunctie (PDF)')

  plt.legend()
  plt.grid(True)
  return plt

def MapLaadPalen():
    #api binnenhalen
    response = requests.get("https://api.openchargemap.io/v3/poi/?output=json&countrycode=NL&maxresults=5000&compact=true&verbose=false&key=09b38fb2-dbc2-408b-94a4-be4480a3022e")
    responsejson=response.json()
    #voorbeeld uit DLO
    df1 = pd.json_normalize(responsejson)
    df4 = pd.json_normalize(df1.Connections)
    df5 = pd.json_normalize(df4[0])
    df5.head()
    ###Bestanden samenvoegen
    df = pd.concat([df1, df5], axis=1)
    #Lijst van kolommen die je wilt verwijderen
    kolommen_te_verwijderen = ['GeneralComments', 'AddressInfo.AddressLine2', 'AddressInfo.ContactTelephone2', 'AddressInfo.ContactEmail',
             'AddressInfo.ContactTelephone1', 'AddressInfo.RelatedURL', 'AddressInfo.ContactTelephone1',
             'OperatorsReference', 'DataProvidersReference', 'AddressInfo.AccessComments', 'UUID', 'Connections',
             'AddressInfo.CountryID', 'AddressInfo.DistanceUnit', 'StatusTypeID',
            'DataProviderID', 'SubmissionStatusTypeID', 'AddressInfo.ID', 'DateLastStatusUpdate', 'OperatorID', 'UsageTypeID'
            ]

    # Kolommen verwijderen uit de DataFrame
    df1 = df1.drop(kolommen_te_verwijderen, axis=1)

    # kolommen omzetten naar 'normale' datetime
    df1['DateLastVerified'] = pd.to_datetime(df1['DateLastVerified'])
    df1['DateCreated'] =      pd.to_datetime(df1['DateCreated'])

    # Pas het datum- en tijdformaat aan
    df1['DateLastVerified']= df1['DateLastVerified'].dt.strftime('%Y-%m-%d %H:%M')
    df1['DateCreated'] =     df1['DateCreated'].dt.strftime('%Y-%m-%d %H:%M')
    #Juiste namen cleanen
    #Noord-Holland
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('Noord Holland', "Noord-Holland")
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('North Holland', "Noord-Holland")
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('Nordholland', "Noord-Holland")
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('NH', "Noord-Holland")
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('North-Holland', "Noord-Holland")
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('Holandia Północna', "Noord-Holland")
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('Noord Holand', "Noord-Holland")
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('Noord-Hooland', "Noord-Holland")

    #Zuid-Holland
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('South Holland', "Zuid-Holland")
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('Zuid Holland', "Zuid-Holland")
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('Zuid-Holland ', "Zuid-Holland")
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('ZH', "Zuid-Holland")
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('Stellendam', "Zuid-Holland")
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('MRDH', "Zuid-Holland")

    #Noord-Brabant
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('North Brabant', "Noord-Brabant")
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('Noord Brabant', "Noord-Brabant")
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('Nordbrabant', "Noord-Brabant")

    #Utrecht
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('UT', "Utrecht")
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('UTRECHT', "Utrecht")
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('UtrechtRECHT', "Utrecht")
    #Zeeland
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('Seeland', "Zeeland")
    #Friesland
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('FRL', "Friesland")
    #Gelderland
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('Stadsregio Arnhem Nijmegen', "Gelderland")
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('Gelderland ', "Gelderland")
    #flevoland
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('Flevolaan', "Flevoland")
    #Overijssel
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('Regio Zwolle', "Overijssel")
    #Drenthe 
    df1['AddressInfo.StateOrProvince'] = df1['AddressInfo.StateOrProvince'].str.replace('Drente', "Drenthe")
    #Verwijder rijen die ontbreken
    df1.dropna(subset=['AddressInfo.Postcode'], inplace=True)
    #df1 = df1[df1['AddressInfo.StateOrProvince'].str.strip() != '']
    df1['AddressInfo.Postcode'] = df1['AddressInfo.Postcode'].str.replace(' ', '')
    df1['AddressInfo.Postcode'] = df1['AddressInfo.Postcode'].str.extract('(\d+)')
    df1.dropna(subset=['AddressInfo.Postcode'], inplace=True)
    df1['AddressInfo.Postcode'] = df1['AddressInfo.Postcode'].astype('int64')
    #extra dataset toevoegen om juiste provincies te krijgen.
    postcode_data1 = pd.read_excel('postcodes.xlsx')

    # Schrijf de inhoud van het Excel-bestand naar een CSV-bestand
    postcode_data1.to_csv('postcodes.csv', index=False)
    # Voeg de provinciegegevens toe aan df
    df1 = df1.merge(postcode_data1, left_on='AddressInfo.Postcode', right_on='Postcode', how='inner')
    # Overschrijf NaN-waarden 
    df1['AddressInfo.Postcode'] = df1['AddressInfo.Postcode'].combine_first(df1['Provincie'])
    # Verwijder de verkeerd opgegeven adressen uit de dataset
    df1 = df1[df1['AddressInfo.Title'] != 'Westervoortsedijk 73-VB']
    df1 = df1[df1['AddressInfo.Title'] != 'Maatheide']
    df1 = df1[df1['AddressInfo.Title'] != 'Informaticalaan 9']
    df1 = df1[df1['AddressInfo.Title'] != 'Tram 1']
    df1 = df1[df1['AddressInfo.Title'] != 'Hanzelaan']
    df1 = df1[df1['AddressInfo.Title'] != 'Amsterdamseweg 53']
    df1 = df1[df1['AddressInfo.Title'] != 'Pedro de Medinalaan 51']
    df1 = df1[df1['AddressInfo.Title'] != 'Plaats van het Journaal']
    df1 = df1[df1['AddressInfo.Title'] != 'Hotel Tiel']
    df1 = df1[df1['AddressInfo.Title'] != 'Floriadepark']
    df1 = df1[df1['AddressInfo.Title'] != 'Sigarenmaker 5']
    df1 = df1[df1['AddressInfo.Title'] != 'Sigarenmaker 8']
    df1 = df1[df1['AddressInfo.Title'] != 'Sigarenmaker 14']
    df1 = df1[df1['AddressInfo.Title'] != 'Duintuin 7']
    df1 = df1[df1['AddressInfo.Title'] != 'Chromiumweg 7']
    df1 = df1[df1['AddressInfo.Title'] != 'Chromiumweg 12']
    #map maken
    # Het gemiddelde als startlocatie
    latitude, longitude = df1['AddressInfo.Latitude'].mean(), df1['AddressInfo.Longitude'].mean()

    # Folium kaart
    m = folium.Map(location=[latitude, longitude], zoom_start=7)

    # Kleuren dictionary voor elke provincie
    kleuren = {
        'Zeeland': '#1f77b4',
        'Drenthe': '#ff7f0e',
        'Noord-Holland': '#2ca02c',
        'Noord-Brabant': '#d62728',
        'Zuid-Holland': '#9467bd',
        'Utrecht': '#8c564b',
        'Limburg': '#e377c2',
        'Friesland': '#7f7f7f',
        'Groningen': '#bcbd22',
        'Overijssel': '#17becf',
        'Flevoland': '#aec7e8 ',
        'Gelderland': '#ffbb78',
    }

    # Maak groep voor elke provincie
    provincie_markers = {}
    for provincie in df1['Provincie'].unique():
        provincie_markers[provincie] = folium.FeatureGroup(name=provincie)

    # Voeg markers toes
    for index, row in df1.iterrows():
        latitude, longitude = row['AddressInfo.Latitude'], row['AddressInfo.Longitude']
        title = row['AddressInfo.Title']
        provincie = row['Provincie']

        #provincie op kleur en als cirkel weergegeven
        folium.CircleMarker([latitude, longitude], radius=3, color=kleuren.get(provincie), fill=True, 
                            fill_color=kleuren.get(provincie, 'gray'), 
                            fill_opacity=0.1, popup=title).add_to(provincie_markers[provincie])

    # Voeg alle features toe aan de kaart
    for provincie, marker_group in provincie_markers.items():
        marker_group.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    #laadpalen per provincie.
    provincie_aantallen = df1['Provincie'].value_counts()

    # Gegeven kleurenpalet
    kleurenpalet = {
        'Zeeland': '#1f77b4',
        'Drenthe': '#ff7f0e',
        'Noord-Holland': '#2ca02c',
        'Noord-Brabant': '#d62728',
        'Zuid-Holland': '#9467bd',
        'Utrecht': '#8c564b',
        'Limburg': '#e377c2',
        'Friesland': '#7f7f7f',
        'Groningen': '#bcbd22',
        'Overijssel': '#17becf',
        'Flevoland': '#aec7e8',
        'Gelderland': '#ffbb78'
    }

    #Staafdiagram
    plt.figure(figsize=(8, 6)) 
    ax = provincie_aantallen.plot(kind='bar', color=[kleurenpalet.get(provincie) for provincie in provincie_aantallen.index])

    #labels
    plt.xlabel('Provincie')
    plt.ylabel('Aantal Laadpalen')
    plt.title('Aantal laadpalen per Provincie')

    # Voeg de aantallen toe 
    for i, v in enumerate(provincie_aantallen):
        ax.text(i, v + 8, str(v), ha='center')

    fig1 = plt

    # Top 10 gemeentes
    top_10_gemeentes = df1['Gemeente'].value_counts().nlargest(10)
    # Gegeven kleurenpalet
    kleurenpalet2 = {
        'Amsterdam': '#2ca02c',
        'Haarlemmermeer': '#2ca02c',
        'Haarlem': '#2ca02c',
        'Rotterdam': '#9467bd',
        'Utrecht': '#8c564b',
        'Alphen aan den Rijn': '#9467bd',
        'Friesland': '#7f7f7f',
        'Groningen': '#bcbd22',
        'Zwolle': '#17becf',
        'Almere': '#aec7e8'
    }

    # Staafdiagram 
    plt.figure(figsize=(8, 6)) 
    ax = top_10_gemeentes.plot(kind='bar', color=[kleurenpalet2.get(gemeente, '#9467bd') for gemeente in top_10_gemeentes.index])

    # Voeg labels toe
    plt.xlabel('Gemeente')
    plt.ylabel('Aantal laadpalen')
    plt.title('Top 10 Gemeentes met de meeste laadpalen')

    # Voeg de aantallen toe
    for i, v in enumerate(top_10_gemeentes):
        ax.text(i, v + 5, str(v), ha='center')

    fig2 = plt
    # Toon de kaart
    return m,fig1,fig2

def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="hello",
    )

    GekentekendeVoertuigenBrandstof = LaadDataAPI('https://opendata.rdw.nl/resource/8ys7-d773.json?$limit=100000')
    GekentekendeVoertuigen = LaadDataAPI('https://opendata.rdw.nl/resource/m9d7-ebf2.json?$limit=100')
    LaadPalen = pd.read_csv('laadpaaldata.csv')
    plot = HistLaadpalen(LaadPalen)
    st.pyplot(fig=plot, clear_figure=None, use_container_width=True)

    KaartPaal,AantalPProv,TopGemeente = MapLaadPalen()
    folium_static(KaartPaal)
    st.write("""In deze dataset staan gegevens over laadpalen in Nederland. Deze gegevens zijn verkregen via een API, waarmee we informatie hebben verzameld over laadpalen verspreid over het hele land. Onze analyse richtte zich op het begrijpen van de verdeling van laadpalen per provincie. We hebben ervoor gekozen om deze informatie te visualiseren doormiddel van een kaart. De punten op de kaart vertegenwoordigen individuele laadpalen en geven aan waar deze zich bevinden in Nederland. Elk van deze punten draagt bij aan het grotere geheel: het netwerk van laadinfrastructuur dat elektrische voertuigen ondersteunt in ons land.
Om de analyse visueel interessanter te maken, hebben we ervoor gekozen om de laadpalen te kleuren op basis van de provincies waarin ze zich bevinden. Deze kleurcodering helpt ons om patronen en verschillen tussen provincies gemakkelijk te herkennen. Het stelt ons in staat om te zien hoe de beschikbaarheid van laadinfrastructuur varieert in verschillende delen van Nederland.

             
             """)
    st.pyplot(fig=AantalPProv, clear_figure=None, use_container_width=True)
    st.write("""De grafiek toont het aantal laadpalen per provincie in Nederland. Het staafdiagram geeft duidelijk weer welke provincies het meeste aantal laadpalen hebben en welke provincies wat achterblijven in de uitrol van laadinfrastructuur voor elektrische voertuigen. Nederland heeft als geheel een sterk ontwikkelde laadinfrastructuur , maar er is nog steeds variatie is tussen provincies. De analyse benadrukt het belang van een evenwichtige uitrol van laadpalen om een breed scala aan gebruikers te bedienen. Uit de grafiek kunnen we afleiden dat "Noord-Holland" de provincie is met het hoogste aantal laadpalen, gevolgd door "Zuid-Holland". Aan de andere kant hebben provincies zoals "Limburg" en "Zeeland" relatief minder laadpalen in vergelijking met de rest van Nederland.
             
             """)
    st.pyplot(fig=TopGemeente, clear_figure=None, use_container_width=True)
    st.write("""De grafiek presenteert een overzicht van de top 10 gemeentes met het hoogste aantal laadpalen in Nederland. Deze analyse werpt licht op de inspanningen van deze gemeentes om elektrische voertuigen te ondersteunen en laat zien hoe ze zich verhouden tot andere gemeentes in het land. 
Kleuren worden gebruikt om elke gemeente in de top 10 gemakkelijk te identificeren. Elke kleur staat in verband met de kleur van de provincies, zo valt te zien dat er drie gemeentes uit zowel Noord- als Zuid-holland in de top 10 staan met meeste laadpalen.
""")


if __name__ == "__main__":
    run()
