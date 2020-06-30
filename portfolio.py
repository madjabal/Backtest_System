# Author: Michael Djaballah
# Last edited: 
# Last edited by: Michael Djaballah

import yfinance as yf
import pandas as pd
from datetime import datetime
import os
from time import sleep, time
from dateutil.relativedelta import relativedelta
import sklearn as sk
from sklearn.preprocessing import StandardScaler

class Portfolio:
    def __init__(self, tickers, hist_depth=None, train_depth=None, features=[], 
                 data_path = 'data/', prefix = 'monthly/', interval = '1mo', 
                 data_start = '2001-01-01', target='Close', download_new=False):
        self.portfolio = {}
        self.tickers = tickers
        self.features = features
        self.target = target
        
        self.hist_depth = hist_depth
        self.train_depth = train_depth
        
        self.interval = interval
        self.data_start = data_start
        
        self.data_path = data_path
        self.prefix = prefix
        
        self.results = []
        self.specific_returns = []
        
        self.blacklist = set()
        
        self.portfolio = {}
        self.build_portfolio(download_new=download_new)
        
    
    def get_data(self, download_new=False, return_bad_tickers=False):
        bad_tickers = []

        os.makedirs(self.data_path + self.prefix, exist_ok=True)

        curr_tickers = set(os.listdir(self.data_path + self.prefix))

        for ticker in self.tickers:
            ticker_label = ticker + '.csv'
            
            if not download_new:
                if ticker_label not in curr_tickers:
                    temp_ticker = yf.Ticker(ticker)
                    temp_hist = temp_ticker.history(start=self.data_start, interval=self.interval)
                    temp_hist.dropna(axis=0, inplace=True)
                    temp_hist.to_csv(self.data_path + self.prefix + ticker_label)

                    if len(temp_hist) < 90:
                        bad_tickers.append((ticker, len(temp_hist)))
                    sleep(.5)
            else:
                temp_ticker = yf.Ticker(ticker)
                temp_hist = temp_ticker.history(start=self.data_start, interval=self.interval)
                temp_hist.dropna(axis=0, inplace=True)
                temp_hist.to_csv(self.data_path + self.prefix + ticker_label)

                if len(temp_hist) < 90:
                    bad_tickers.append((ticker, len(temp_hist)))
                sleep(.5)

        if return_bad_tickers:
            return bad_tickers

        return None
    
    
    def check_ticker(self, ticker, offset):
        ticker_df = pd.read_csv(self.data_path + self.prefix + ticker + '.csv')
        if len(ticker_df) >= offset:
            return ticker_df
        return False
    
    
    def build_portfolio(self, download_new=False):
        offset = self.train_depth + self.hist_depth + 60 + 6

        self.get_data(download_new=download_new)

        ticker_dict = {}

        for ticker in self.tickers:
            if ticker not in self.blacklist:
                ticker_df = self.check_ticker(ticker, offset)
                if type(ticker_df) != bool:
                    ticker_dict[ticker] = ticker_df
        
        self.tickers = list(ticker_dict.keys())
        self.portfolio = ticker_dict
        return ticker_dict
    
    
    def build_feature_vector(self, ticker, date, keep_pred=True):
        ticker_df = self.portfolio[ticker]

        start_date_dt = datetime.strptime(date, '%Y-%m-%d') - relativedelta(months=self.hist_depth)
        start_date = start_date_dt.strftime('%Y-%m-%d')

        feature_df = ticker_df.set_index('Date')[start_date:date].reset_index(drop=True)[self.features]

        new_df_dict = {}

        for i in range(len(feature_df)):
            for col in feature_df.columns:
                if i < len(feature_df) - 1:
                    new_df_dict[col + ' ' + str(i + 1)] = [feature_df[col].iloc[i]]
                elif col == self.target:
                    if keep_pred:
                        new_df_dict['Target'] = [feature_df[col].iloc[i]]

        new_df = pd.DataFrame.from_dict(new_df_dict)
        
        if len(new_df) == 0:
            self.blacklist.add(ticker)
            return -1

        if keep_pred:
            new_df = new_df[[col for col in list(new_df.columns) if col not in {'Target'}] + ['Target']]
        
        return new_df
    
    
    def build_train_df(self, date):    
        vector_list = []
        for ticker in self.tickers:
            if ticker not in self.blacklist:
                for i in range(self.train_depth):
                    train_start_dt = datetime.strptime(date, '%Y-%m-%d') - relativedelta(months=(1+i))
                    train_start = train_start_dt.strftime('%Y-%m-%d')
                    vector = self.build_feature_vector(ticker, train_start)
                    
                    if type(vector) != int:
                        vector_list.append(vector)
        
        feature_df = pd.concat(vector_list)
        return feature_df.reset_index(drop=True)
    
    
    def build_test_df(self, date):
        vector_list = []
        index_list = []
        for ticker in self.tickers:
            if ticker not in self.blacklist:
                vector = self.build_feature_vector(ticker, date, keep_pred=False)
                if type(vector) != int:
                    vector_list.append(vector)
                    index_list.append(ticker)

        test_df = pd.concat(vector_list)
        return test_df.reset_index(drop=True), index_list
    
    
    def build_returns(self, symbols, date):
        returns = []
        for ticker in symbols:
            temp_ticker_dict = self.portfolio[ticker].set_index('Date').loc[date]
            returns.append((temp_ticker_dict['Close'] - temp_ticker_dict['Open'])/temp_ticker_dict['Open'])
        return returns
    
    
    def build_scaled_df(self, dataframe):
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(dataframe)
        scaled_dataframe = pd.DataFrame(scaled_array, columns=dataframe.columns)
        return scaled_dataframe
    
    
    def check_date(self, ticker, date):
        dates = set(self.portfolio[ticker]['Date'])
        return date in dates
    
    
    def build_machine(self, model, date):
        train_df = self.build_train_df(date)
        scaled_train_df = self.build_scaled_df(train_df)

        scaled_train_df.dropna(axis=0, inplace=True)

        X = scaled_train_df.values[:,:-1]
        y = scaled_train_df.values[:, -1]
        model.fit(X, y)

        test_df, symbols = self.build_test_df(date)
        scaled_test_df = self.build_scaled_df(test_df)
        X_test = scaled_test_df.values

        predicted_returns = list(model.predict(X))

        returns_dict = {}

        for i in range(len(symbols)):
            returns_dict[symbols[i]] = predicted_returns[i]

        top = sorted(returns_dict.items(), key=lambda x: x[1])[::-1]
        return [x[0] for x in top]
    
    
    def backtest(self, model, start_date, end_date, n=15, m=0):
        months = list(pd.date_range(start_date, end_date, freq='MS').strftime('%Y-%m-%d'))

        overall_returns = []
        for month in months:
            start_time = time()
            for ticker in self.tickers:
                if ticker not in self.blacklist:
                    if not self.check_date(ticker, month):
                        self.blacklist.add(ticker)
            
            symbols = self.build_machine(model, month)
            
            long_ticker_returns = self.build_returns(symbols[:n], month)
            if m != 0:
                short_ticker_returns = self.build_returns(symbols[-1 * m:], month)
            else:
                short_ticker_returns = []
            
            long_and_short_returns = sum(long_ticker_returns) + (-1 * sum(short_ticker_returns))
            long_and_short_returns /= (n + m)
            
            overall_returns.append(long_and_short_returns)
            
            print(month, round(long_and_short_returns, 6), round(time()-start_time, 2))
            
            specific_returns_dict = {'LONG': {}, 'SHORT': {}}
            for i in range(len(long_ticker_returns)):
                specific_returns_dict['LONG'][symbols[i]] = long_ticker_returns[i]
            
            for i in range(len(short_ticker_returns)):
                specific_returns_dict['SHORT'][symbols[i]] = short_ticker_returns[i]
            
            self.specific_returns.append(specific_returns_dict)
        
        self.returns = overall_returns
        return overall_returns
    
    

def get_snp_store(data_path='data/'):
    curr_raw = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    
    curr = curr_raw[0]
    hist = curr_raw[1]
    
    new_hist = pd.DataFrame(hist['Date'])
    new_hist['Added'] = hist['Added', 'Ticker']
    new_hist['Removed'] = hist['Removed', 'Ticker']
    
    os.makedirs(data_path, exist_ok=True)
    
    curr.to_csv(data_path + 'snp_current.csv', index=False)
    new_hist.to_csv(data_path + 'snp_hist.csv', index=False)
    return None


def build_snp(date, data_path='data/'):
    curr = pd.read_csv(data_path + 'snp_current.csv')
    hist = pd.read_csv(data_path + 'snp_hist.csv')
    
    start_date = datetime.strptime(date, '%Y-%m-%d')
    
    snp_set = set(curr['Symbol'])
    
    for i in range(len(hist)):
        temp_date = datetime.strptime(hist.iloc[i]['Date'], '%Y-%m-%d')
        if temp_date < start_date:
            break

        tb_removed = hist.iloc[i]['Added']
        tb_added = hist.iloc[i]['Removed']

        if tb_removed in snp_set:
            snp_set.remove(tb_removed)
        if not type(tb_added) == float:
            snp_set.add(tb_added)
    
    return list(snp_set)