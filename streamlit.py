from PIL import Image
from model.ta_svm import get_new_data, train_new_model
import streamlit as st
import plotly.graph_objects as go
import fear_and_greed


def get_fear_and_greed_value():
    fear_and_greed_value = fear_and_greed.get().value
    return fear_and_greed_value


def create_gauge_chart(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': 'darkblue'},
            'steps': [
                {'range': [0, 25], 'color': 'red'},
                {'range': [25, 50], 'color': 'orange'},
                {'range': [50, 75], 'color': 'yellow'},
                {'range': [75, 100], 'color': 'green'}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': value}
        },
        title={'text': "Fear & Greed Index"}
    ))
    return fig


def create_candlestick_chart(df, start_time, end_time, min_price, max_price):

    predictions = train_new_model(df)
    # Create a new column in the DataFrame with the predictions
    df['prediction'] = predictions
    df['prediction'] = df['prediction'].map({1: 'up', 0: 'down'})
    df['prediction'] = df['prediction'].shift(-1)

    # Replace the dummy prediction logic with the actual predictions from the SVM model
    # Create scatter plot trace for up arrows
    up_arrows = go.Scatter(
        x=df[df['prediction'] == 'up']['time'],
        y=df[df['prediction'] == 'up']['low'] -
        (max_price - min_price) * 0.05,  # Adjust the position of the arrow
        mode='markers',
        marker=dict(
            symbol='triangle-up',
            size=10,
            color='green'
        ),
        name='Up Prediction'
    )

    # Create scatter plot trace for down arrows
    down_arrows = go.Scatter(
        x=df[df['prediction'] == 'down']['time'],
        y=df[df['prediction'] == 'down']['high'] +
        (max_price - min_price) * 0.05,  # Adjust the position of the arrow
        mode='markers',
        marker=dict(
            symbol='triangle-down',
            size=10,
            color='red'
        ),
        name='Down Prediction'
    )

    fig = go.Figure()

    # Add candlestick trace
    fig.add_trace(go.Candlestick(x=df['time'],
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 name='Candlestick'))

    # Add up and down arrow traces
    fig.add_trace(up_arrows)
    fig.add_trace(down_arrows)

    fig.update_layout(title='Candlestick Chart',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False,
                      xaxis=dict(range=[start_time, end_time]),
                      yaxis=dict(range=[min_price, max_price]),
                      dragmode='pan',
                      width=1500,  # Set the width of the chart
                      height=1000)  # Set the height of the chart

    return fig


df = get_new_data()


st.set_page_config(layout="wide")

image = Image.open('assets/logo.png')

# Display the last 50 candles by default
default_candles = 25
start_index = len(df) - default_candles
end_index = len(df)
start_time = df.iloc[start_index]['time']
end_time = df.iloc[-1]['time']

# Calculate min and max price for the visible range
min_price = df.iloc[start_index:end_index]['low'].min()
max_price = df.iloc[start_index:end_index]['high'].max()

chart = create_candlestick_chart(
    df, start_time, end_time, min_price, max_price)

fear_and_greed_value = get_fear_and_greed_value()
gauge_chart = create_gauge_chart(fear_and_greed_value)

# Create columns for the candlestick chart and the gauge chart with the image above it
col1, col2 = st.columns([2, 1])  # Adjust the column width ratio
with col1:
    st.plotly_chart(chart)
with col2:
    st.image(image, width=600)
    st.plotly_chart(gauge_chart)
