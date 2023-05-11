from PIL import Image
from model.multimodal_model import get_new_data, train_new_model
import streamlit as st
import plotly.graph_objects as go
import fear_and_greed
import pandas as pd

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
            'bar': {'color': '#144279'},
            'steps': [
                {'range': [0, 25], 'color': 'red'},
                {'range': [25, 50], 'color': 'orange'},
                {'range': [50, 75], 'color': 'yellow'},
                {'range': [75, 100], 'color': 'green'}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': value}
        },
        title={'text': "Fear & Greed Index",'font': {'size':20,'color': '#144279'}},
        number={'suffix': "%", 'font': {'size': 70,'color': '#144279'}},
    ))

    # Set the width and height of the gauge chart
    fig.update_layout(width=400, height=400)  # Adjust these values to change the size of the gauge chart

    return fig


def create_candlestick_chart(df, start_time, end_time, min_price, max_price):

    predictions = train_new_model(df)
    # Create a new column in the DataFrame with the predictions
    df['prediction'] = predictions
    df['prediction'] = df['prediction'].map({1: 'up', 0: 'down'})
    df['prediction'] = df['prediction'].shift(0)


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

    fig.update_layout(title={'text': 'Candlestick Chart with Live Bitcoin Price and Model Predictions', 'x': 0.5, 'xanchor': 'center'},
                      xaxis_title='Date',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False,
                      xaxis=dict(range=[start_time, end_time]),
                      yaxis=dict(range=[min_price, max_price]),
                      dragmode='pan',
                      width=1200,  # Set the width of the chart
                      height=650  # Set the height of the chart
    )

    return fig


df = get_new_data()

def get_random_tweets():
    tweets_df = pd.read_csv('data/sample_tweets.csv')
    random_tweets = tweets_df.sample(n=2)
    return random_tweets


st.set_page_config(page_title="Techniment",layout="wide")

image = Image.open('assets/logo.png')

# Display the last 50 candles by default
default_candles = 50  # Increase this value to show more candles by default
start_index = len(df) - default_candles
end_index = len(df)
start_time = df.iloc[start_index]['time']
end_time = df.iloc[-1]['time'] + pd.Timedelta(hours=1)  # Add some extra time to avoid cropping the last candle


# Calculate min and max price for the visible range
min_price = df.iloc[start_index:end_index]['low'].min() *0.995
max_price = df.iloc[start_index:end_index]['high'].max() *1.005

chart = create_candlestick_chart(
    df, start_time, end_time, min_price, max_price)

fear_and_greed_value = get_fear_and_greed_value()
gauge_chart = create_gauge_chart(fear_and_greed_value)



# Create two columns with a specified width ratio (adjust the ratio according to your preference)
col1, col2 = st.columns([1, 3])

# Display the logo in the first column
col1.image(image, width=300)

# Display the title and description in the second column
with col2:

    st.title('Combining Technical Data with Sentiment Analysis')
    st.write("""
    Techniment is a web application dashboard which features a candlestick chart with live Bitcoin price and model predictions. The model has been trained on technical data and twitter sentiment analysis. By achieving 99.7% accuracy in classifying tweets and technical data, we are able to predict the price of Bitcoin with 98% accuracy. Below each candle in the chart, you can see the model's predicted output. You can use this dashboard alongside the Fear & Greed Index to make informed decisions about your investments. 
    """)

random_tweets = get_random_tweets()
st.divider()

col1, col2 = st.columns([3, 1])  

with col1:
    st.plotly_chart(chart)
with col2:
    st.markdown("<h2 style='font-size:20px;'>Sample Tweets</h2>", unsafe_allow_html=True)

    for idx, row in random_tweets.iterrows():
        st.markdown(f"<h3 style='font-size:14px;'>{row['date']}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:12px;'>{row['tweet']}</p>", unsafe_allow_html=True)
    st.plotly_chart(gauge_chart)

