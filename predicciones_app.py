import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------
# Funciones de predicción
# -----------------------

@st.cache_data
def cargar_datos(excel_path):
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()
    df['FECHA_VENTA'] = pd.to_datetime(df['FECHA_VENTA'])
    df = df.sort_values(by='FECHA_VENTA')
    return df

def entrenar_prophet(df, periodo):
    df_p = df[['FECHA_VENTA', 'CANTIDAD_VENDIDA']].rename(columns={'FECHA_VENTA': 'ds', 'CANTIDAD_VENDIDA': 'y'})
    model = Prophet()
    model.fit(df_p)
    future = model.make_future_dataframe(periods=periodo)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].set_index('ds')

def entrenar_arima(df, periodo):
    serie = df.set_index('FECHA_VENTA')['CANTIDAD_VENDIDA']
    modelo = ARIMA(serie, order=(5,1,0))
    modelo_fit = modelo.fit()
    pred = modelo_fit.forecast(steps=periodo)
    fechas = pd.date_range(start=df['FECHA_VENTA'].max() + pd.Timedelta(days=1), periods=periodo)
    return pd.Series(pred.values, index=fechas)

def entrenar_rnn(df, periodo):
    look_back = 5
    ventas = df['CANTIDAD_VENDIDA'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    ventas_norm = scaler.fit_transform(ventas)

    X, y = [], []
    for i in range(len(ventas_norm) - look_back):
        X.append(ventas_norm[i:i+look_back])
        y.append(ventas_norm[i+look_back])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=20, batch_size=1, verbose=0)

    ultimos = X[-1]
    predicciones = []
    for _ in range(periodo):
        pred = model.predict(ultimos.reshape(1, look_back, 1), verbose=0)
        predicciones.append(pred[0, 0])
        ultimos = np.append(ultimos[1:], [[pred[0, 0]]], axis=0)

    pred_final = scaler.inverse_transform(np.array(predicciones).reshape(-1,1)).flatten()
    fechas = pd.date_range(start=df['FECHA_VENTA'].max() + pd.Timedelta(days=1), periods=periodo)
    return pd.Series(pred_final, index=fechas)

# -----------------------
# Interfaz Streamlit
# -----------------------

st.title("Predicción de Demanda de Inventario")

excel_path = "Items_Morante.xlsx"
df = cargar_datos(excel_path)
periodo = st.slider("Días a predecir", min_value=7, max_value=60, value=45)

items = df['ITEM'].unique()
st.subheader("Predicción rápida para los primeros 20 ítems")

primeros_items = df['ITEM'].unique()[:20]

for i in primeros_items:
    st.markdown(f"### Ítem: {i}")
    df_i = df[df['ITEM'] == i].copy()
    
    try:
        prophet_pred_i = entrenar_prophet(df_i, periodo)
        arima_pred_i = entrenar_arima(df_i, periodo)
        rnn_pred_i = entrenar_rnn(df_i, periodo)
        real_i = df_i.set_index('FECHA_VENTA')['CANTIDAD_VENDIDA'][-periodo:]

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(real_i.index, real_i.values, label='Real', marker='o')
        ax.plot(prophet_pred_i[-periodo:].index, prophet_pred_i[-periodo:]['yhat'], label='Prophet', marker='x')
        ax.plot(arima_pred_i.index, arima_pred_i.values, label='ARIMA', marker='s')
        ax.plot(rnn_pred_i.index, rnn_pred_i.values, label='RNN', marker='d')
        ax.set_title(f'Predicción para {i}')
        ax.legend(fontsize="small")
        st.pyplot(fig)
    except:
        st.warning(f"No se pudo generar predicción para el ítem {i}.")


item_seleccionado = st.selectbox("Selecciona un ítem para analizar:", items)

df_item = df[df['ITEM'] == item_seleccionado].copy()
descripcion = df_item['DESCRIPCION'].iloc[0]
st.write(f"**Descripción del ítem:** {descripcion}")


# Predicciones
prophet_pred = entrenar_prophet(df_item, periodo)
arima_pred = entrenar_arima(df_item, periodo)
rnn_pred = entrenar_rnn(df_item, periodo)

# Real
real = df_item.set_index('FECHA_VENTA')['CANTIDAD_VENDIDA'][-periodo:]

# Visualización
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(real.index, real.values, label='Real', marker='o')
ax.plot(prophet_pred[-periodo:].index, prophet_pred[-periodo:]['yhat'], label='Prophet', marker='x')
ax.plot(arima_pred.index, arima_pred.values, label='ARIMA', marker='s')
ax.plot(rnn_pred.index, rnn_pred.values, label='RNN', marker='d')
ax.set_title(f'Predicción de ventas para {item_seleccionado}')
ax.legend()
st.pyplot(fig)

# MAE
st.subheader("Evaluación MAE")
st.write(f"**Prophet:** {mean_absolute_error(real.values, prophet_pred[-periodo:]['yhat'].values):.2f}")
st.write(f"**ARIMA:** {mean_absolute_error(real.values, arima_pred.values):.2f}")
st.write(f"**RNN:** {mean_absolute_error(real.values, rnn_pred.values):.2f}")
