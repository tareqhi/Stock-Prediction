# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date
from bdshare import get_hist_data
#import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')
tickerDf = get_hist_data('2021-05-12','2021-05-12')
sorted_unique_symbol = sorted(tickerDf.symbol.unique())
selected_stock = st.sidebar.selectbox('Symbol', sorted_unique_symbol)

n_years = st.sidebar.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
	data = get_hist_data(START, TODAY, ticker)
	data.reset_index(inplace=True)
	return data

#data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
#data_load_state.text('Loading data... done!')

st.subheader('Source data')
st.write(data)

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['date'], y=data['open'], name="Openning Price"))
	fig.add_trace(go.Scatter(x=data['date'], y=data['close'], name="Closing Price"))
	fig.layout.update(title_text='Visual chart of Openning & Closing price!', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['date','close']]
df_train = df_train.rename(columns={"date": "ds", "close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data\n***')
st.write(f'Forecast plot for {n_years} years')
#st.write(forecast.tail())

fig1 = plot_plotly(m, forecast)
components_fig = m.plot_components(forecast)
axes = components_fig.get_axes()
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Price')
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)