# RLAgent_TradingStocks
This repository consists of a Deep Reinforcement Learning Agent for Trading Stocks, implemented on Apple stocks from the US S&amp;P 500 Dataset.

## Main Idea

This project involves a Deep Q-Learning Reinforcement Learning Model to estimate the optimal actions of a trader, and return the profit value incurred on a dataset.
The model uses an experience replay and Double DQN with input features given by the current state of the stock, comprising 33 technical indicators, and available actions, namely buying, selling and holding a stock, while the output is the Q-value function estimating the future rewards. 


#### Presentation Slides

Further details regarding the methods and results of implementation can be found in the uploaded pdf document.

### Data

#### Acquisition

The data was acquired through the publicl available [**US S&P 500 dataset.**](https://github.com/yumoxu/stocknet-dataset) Only the prices of the apple stocks were considered.

#### Feature Generation

Technical indicators are derived from fundamental price and volume in the categories of:
* Trend
* Momentum
* Volatility
* Volume

The data has a total of ***33 technical features*** and is then normalized and fed through the Double DQN

#### Training Data

The RL agent is trained on 5 years of historical data.

#### Test Data

The RL agent is tested on an unseen set of 8 months of price/volume data.

### Model

1. The Agent observes the environment and take notes of the state
2. Based on that state, the Agent takes an action based on a policy.
3. The Agent therefore receives a reward from the environment.
4. The environment transitions to a new state based on the action by the Agent.
5. Repeat

Instead of storing a massive lookup table, this project will approximate Q(s,a) with neural networks, namely a Deep Q Network (DQN)
