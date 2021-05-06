import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import itertools
import argparse
import re
import os
import pickle
import random
from sklearn.preprocessing import StandardScaler
from trend import  get_data, get_trend
from sentiment import get_sentiment
from stocks import get_stock_value, get_stock_value_multi
from datetime import timedelta
        
        
class LinearModel:
    """ A linear regression model """
    def __init__(self, input_dim, n_action):
        self.W = np.random.randn(input_dim, n_action) / np.sqrt(input_dim)
        self.b = np.zeros(n_action)
        self.vW = 0
        self.vb = 0
        self.losses = []

    def predict(self, X):
        assert(len(X.shape) == 2)
        return X.dot(self.W) + self.b

    def sgd(self, X, Y, learning_rate=0.01, momentum=0.9):
        assert(len(X.shape) == 2)
        num_values = np.prod(Y.shape)
        Yhat = self.predict(X)
        gW = 2 * X.T.dot(Yhat - Y) / num_values
        gb = 2 * (Yhat - Y).sum(axis=0) / num_values
        # update momentum terms
        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb
        # update params
        self.W += self.vW
        self.b += self.vb
        mse = np.mean((Yhat - Y)**2)
        self.losses.append(mse)
        
    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']

    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)

        
class MultiStockEnv:
    """
    A 3-stock trading environment.
    State: vector of size 7 (n_stock * 2 + 1)
    - # shares of stock 1 owned
    - # shares of stock 2 owned
    - # shares of stock 3 owned
    - price of stock 1 (using daily close price)
    - price of stock 2
    - price of stock 3
    - cash owned (can be used to purchase more stocks)
    Action: categorical variable with 27 (3^3) possibilities
    - for each stock, you can:
    - 0 = sell
    - 1 = hold
    - 2 = buy
    """
    def __init__(self, stock_data, senti_data, trend_data, initial_investment=20000, predict =False):
        # data
        self.stock_price_history = stock_data
        self.senti_history = senti_data
        self.trend_history = trend_data
        self.n_step, self.n_stock = self.stock_price_history.shape

        # instance attributes
        self.initial_investment = initial_investment
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None
        self.trend = None
        self.senti = None
        self.action_space = np.arange(3**self.n_stock)

        # action permutations
        # returns a nested list with elements like:
        # [0,0,0]
        # [0,0,1]
        # [0,0,2]
        # [0,1,0]
        # [0,1,1]
        # etc.
        # 0 = sell
        # 1 = hold
        # 2 = buy
        self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

        # calculate size of state
        # number of stock owned, stock price, sentiment, trend, cash in hand
        self.state_dim = self.n_stock * 4 + 1 
        self.predict = predict
        self.reset()


    def reset(self):
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.cur_step]
        self.senti = self.senti_history[self.cur_step]
        self.trend = self.trend_history[self.cur_step]
        self.cash_in_hand = self.initial_investment
        return self._get_obs()


    def step(self, action):
        assert action in self.action_space
        # get current value before performing the action
        prev_val = self._get_val()
        # update price, i.e. go to the next day
        self.cur_step = 0
        self.stock_price = self.stock_price_history[self.cur_step]
        self.senti = self.senti_history[self.cur_step]
        self.trend = self.trend_history[self.cur_step]
        # perform the trade
        self._trade(action)
        # get the new value after taking the action
        cur_val = self._get_val()
        # reward is the increase in porfolio value
        reward = cur_val - prev_val
        # done if we have run out of data
        done = self.cur_step == self.n_step - 1
        # store the current value of the portfolio here
        info = {'cur_val': cur_val}
        # conform to the Gym API
        return self._get_obs(), reward, done, info


    def _get_obs(self):
        obs = np.empty(self.state_dim)
        obs[:self.n_stock] = self.stock_owned
        obs[self.n_stock:2*self.n_stock] = self.stock_price
        obs[2*self.n_stock:3*self.n_stock] = self.senti
        obs[3*self.n_stock:4*self.n_stock] = self.trend
        obs[-1] = self.cash_in_hand
        return obs



    def _get_val(self):
        return self.stock_owned.dot(self.stock_price) + self.cash_in_hand


    def _trade(self, action):
        # index the action we want to perform
        # 0 = sell
        # 1 = hold
        # 2 = buy
        # e.g. [2,1,0] means:
        # buy first stock
        # hold second stock
        # sell third stock
        action_vec = self.action_list[action]

        # determine which stocks to buy or sell
        sell_index = [] # stores index of stocks we want to sell
        buy_index = [] # stores index of stocks we want to buy
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)

        # sell any stocks we want to sell
        # then buy any stocks we want to buy
        if sell_index:
          # NOTE: to simplify the problem, when we sell, we will sell ALL shares of that stock
          for i in sell_index:
            self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
            self.stock_owned[i] = 0
        if buy_index:
          # NOTE: when buying, we will loop through each stock we want to buy,
          #       and buy one share at a time until we run out of cash
                can_buy = True
                while can_buy:
                    for i in buy_index:
                        if self.cash_in_hand > self.stock_price[i]:
                            self.stock_owned[i] += 1 # buy one share
                            self.cash_in_hand -= self.stock_price[i]
                        else:
                            can_buy = False


class DQNAgent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = LinearModel(state_size, action_size)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
              return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action


    def train(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.amax(self.model.predict(next_state), axis=1)

        target_full = self.model.predict(state)
        target_full[0, action] = target

        # Run one training step
        self.model.sgd(state, target_full)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, name):
        self.model.load_weights(name)


    def save(self, name):
        self.model.save_weights(name)





def trade(start_date, end_date, investment, company_name, stock_ticker):
    days = 1
    portfolio_value_list, rewards_list, cash_in_hand_list= [], [], []
    cur_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    if end_date.weekday() in [5, 6]:
        end_date = end_date - timedelta(days = end_date.weekday()%4)
    if cur_date.weekday() in [5, 6]:
        cur_date = cur_date + timedelta(days = 7 - end_date.weekday())
    models_folder = 'linear_rl_trader_models'
    state_size = len(company_name) * 4 + 1
    action_size = len(company_name) ** 3
    agent = DQNAgent(state_size, action_size)
    with open(f'{models_folder}/'+company_name[0]+'_scaler.pkl', 'rb') as f:
              scaler = pickle.load(f)
    agent.epsilon = 0.01 
    agent.load(f'{models_folder}/'+company_name[0]+'_linear.npz')
    while cur_date < end_date:
        if cur_date.weekday() not in [5, 6]:
            start = str(cur_date.year) +"-" + str(cur_date.month) +"-" +str(cur_date.day)
            next_day = cur_date + timedelta(days = 1)
            end = str(next_day.year) +"-" + str(next_day.month) +"-" +str(next_day.day)
            senti = get_sentiment(start, end, company_name[0])
            senti = round(senti, 2)
            stock_data = get_data(start, end, stock_ticker[0])
            trend = get_trend(stock_data, company_name[0])
            if days == 1:
                env = MultiStockEnv(np.array([[stock_data['Adj Close']]]), np.array([[senti]]), np.array([[trend]]), investment, True)
                state = env.reset()
                state = scaler.transform([state])
                env.cur_step = 0
            else:
                np.append(env.stock_price_history,np.array([[stock_data['Adj Close']]]))
                np.append(env.senti_history, np.array([[senti]]))
                np.append(env.trend_history, np.array([[trend]]))
                env.n_step += 1
                env.cur_step = env.n_step - 1
            action = agent.act(state) # get action
            next_state, reward, done, info = env.step(action) # get the next state, reward etc 
            next_state = scaler.transform([next_state])
            state = next_state
            portfolio_value_list.append(info['cur_val'])
            rewards_list.append(reward)
            cash_in_hand_list.append(env.cash_in_hand)
        days += 1
        cur_date = cur_date + timedelta(days = 1)
    if len(company_name) > 1:
        returns = get_stock_value_multi(days, investment)
    else:
        returns = get_stock_value(days, investment)    
    print(returns)
    
    return returns


#trade("2021-03-15","2021-03-22",20000,["amazon"],["AMZN"])

