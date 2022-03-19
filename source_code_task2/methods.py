import os
import logging
import numpy as np
from tqdm import tqdm
from source_code_task2.utils import get_state, format_currency, format_position, normalize
import pdb

'''
1. Move daily_pct_return to utils
2. Move calc_reward to utils
'''

def daily_pct_change(prices, shift):
  pct_change = (prices.copy() / prices.copy().shift(periods = shift)) - 1
  pct_change[:shift] = 0
  return pct_change

def calc_reward(pct_change, net_holdings):
  return pct_change * net_holdings

def train_model(agent, episode, data, episode_count = 50, batch_size = 32, window_size = 10):
  total_profit = 0
  num_observations = len(data)

  agent.inventory = []
  shares_history = []
  average_loss = []

  net_holdings = 0
  normed_data = normalize(data)
  pct_change = daily_pct_change(data.price, window_size)

  for t in tqdm(range(num_observations), total = num_observations, leave = True, desc = f'Episode {episode}/{episode_count}'):
    done = t == (num_observations - 1)

    state, time = get_state(normed_data, t)
    action = agent.action(state)

    if action == 2 and net_holdings == 0:
        shares = -100
        net_holdings += -100
    elif action == 2 and net_holdings == 100:
        shares = -200
        net_holdings += -200
    elif action == 1 and net_holdings == 0:
        shares = 100
        net_holdings += 100
    elif action == 1 and net_holdings == -100:
        shares = 200
        net_holdings += 200
    else:
        shares = 0
    shares_history.append(shares)

    reward = calc_reward(pct_change[t] * 100, net_holdings)
    total_profit += reward

    if not done:
      next_state, next_time = get_state(normed_data, t + 1)
      agent.remember(state, time, action, reward, next_state, next_time, done)
      state = next_state
      time = next_time


    if len(agent.memory) > batch_size:
      loss = agent.replay(batch_size)
      average_loss.append(loss)

    # if(episode%10==0):
    #   agent.save(episode)

    if done: return (episode, episode_count, total_profit, np.array(average_loss).mean())

def evaluate_model(agent, data, verbose, window_size = 10):
  total_profit = 0
  num_observations = len(data)

  shares = []
  history = []
  agent.inventory = []
  normed_data = normalize(data)
  cum_return = []
  net_holdings = 0
  shares_history = []
  pct_change = daily_pct_change(data.price, 10)

  for t in range(num_observations):
    done = t == (num_observations - 1)
    reward = 0

    state, time = get_state(normed_data, t)
    action = agent.action(state, evaluation = True)

    if action == 2 and net_holdings == 0:
      shares = -10
      net_holdings += -10
      history.append((data.price[t], "SELL"))
    elif action == 2 and net_holdings == 10:
      shares = -20
      net_holdings += -20
      history.append((data.price[t], "SELL"))
    elif action == 1 and net_holdings == 0:
      shares = 10
      net_holdings += 10
      history.append((data.price[t], "BUY"))
    elif action == 1 and net_holdings == -10:
      shares = 20
      net_holdings += 20
      history.append((data.price[t], "BUY"))
    else:
      shares = 0
      history.append((data.price[t], "HOLD"))
    shares_history.append(shares)

    reward = calc_reward(pct_change[t], net_holdings)
    total_profit += reward

    if not done:
      next_state, next_time = get_state(normed_data, t + 1)
      agent.memory.append((state, time, action, reward, next_state, next_time, done))
      state = next_state
      time = next_time

    if done: return total_profit, history, shares_history


