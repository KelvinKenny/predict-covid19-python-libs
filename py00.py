import re 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime

url = 'https://github.com/neylsoncrepalde/projeto_eda_covid/blob/master/covid_19_data.csv?raw=true'

df = pd.read_csv(url, parse_dates=['ObservationDate', 'Last Update'])
df

df.info()

df.describe().drop(columns='SNo').T

df.columns = [re.sub(r'[/ ]', '_', col).lower() for col in df.columns]
df.sample(4)

df.loc[df.country_region == 'Brazil'].notnull().sum()

df_brasil = df.loc[df.country_region == 'Brazil'].drop(columns=['province_state','sno'])
df_brasil

df_brasil = df_brasil[df_brasil.confirmed > 0]
df_brasil.shape

def dif(v):
    J=[v[i+1]-v[i] for i in range(len(v)-1)]
    J.insert(0, v[0])
    return J

def dif2(v):
    J=[v[0]]
    for i in range(len(v)-1):
        J.append(v[i+1]-v[i])
    return np.array(J)

df_brasil = df_brasil.assign( novoscasos=dif(df_brasil['confirmed'].values) )
df_brasil

from statsmodels.tsa.seasonal import seasonal_decompose
def Grafic_decompose(serie):
    res = seasonal_decompose(serie)

    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(8,8))
    ax1.plot(res.observed)
    ax1.set_title('Série original')

    ax2.plot(res.trend)
    ax2.set_title('Tendência', pad=5)

    ax3.plot(res.seasonal)
    ax3.set_title('Sazonalidade')

    ax4.scatter(novoscasos.index, res.resid)
    ax4.plot(novoscasos.index, res.resid)
    ax4.axhline(0, linestyle='dashed', c='black')
    ax4.set_title('Resíduos')

    fig.tight_layout(pad=0.7)
    plt.show()

    
novoscasos = df_brasil.novoscasos
novoscasos.index = df_brasil.observationdate

Grafic_decompose(novoscasos)

confirmados = df_brasil.confirmed
confirmados.index = df_brasil.observationdate

Grafic_decompose(confirmados)

from pmdarima.arima import auto_arima
modelo = auto_arima(confirmados)
fig = go.Figure(go.Scatter(
    x=confirmados.index, y=confirmados, name='Observed', mode='lines+markers'
))
fig.add_trace(go.Scatter(x=confirmados.index, y = modelo.predict_in_sample(), name='Predicted'))

data_inicial=pd.to_datetime('2020-05-20')
data_final  =pd.to_datetime('2020-06-04')
n = (data_final - data_inicial).days
fig.add_trace(go.Scatter(x=pd.date_range(data_inicial, data_final), y=modelo.predict(n), name='Forecast'))

fig.update_layout(title=f'Previsão de casos confirmados para os próximos {n} dias',
                  yaxis_title='Casos confirmados', xaxis_title='Data',
                  margin=dict(l=30, r=30, t=50, b=5),
                  width=1100, height=400, font=dict(size=14))
fig.show()


