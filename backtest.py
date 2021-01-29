
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
import os
from time import sleep, time
from dateutil.relativedelta import relativedelta
import sklearn as sk
from sklearn.preprocessing import StandardScaler


def build_snp(date, data_path='data/'):
    
    date_format='%Y-%m-%d'
    date_format2 = '%B %d, %Y'
    
    curr = pd.read_csv(data_path + 'snp_current.csv')
    hist = pd.read_csv(data_path + 'snp_hist.csv')
    
    start_date = datetime.strptime(date, date_format)
    
    snp_set = set(curr['Symbol'])
    
    for i in range(len(hist)):
        temp_date = datetime.strptime(hist.iloc[i]['Date'], date_format2)
        if temp_date < start_date:
            break

        tb_removed = hist.iloc[i]['Added']
        tb_added = hist.iloc[i]['Removed']

        if tb_removed in snp_set:
            snp_set.remove(tb_removed)
        if not type(tb_added) == float:
            snp_set.add(tb_added)
    
    return list(snp_set)


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


def get_data(tickers, start='2000-01-01', data_path='data/', get_new=False):
    interval = '1d'
    prefix = interval + '/'
    
    os.makedirs(data_path + prefix, exist_ok=True)
    
    if get_new:
        curr_tickers = set()
    else:
        curr_tickers = set(os.listdir(data_path + prefix))
    
    for ticker in tickers:
        ticker_label = ticker + '.csv'
        
        if ticker_label not in curr_tickers:
            temp_ticker = yf.Ticker(ticker)
            temp_hist = temp_ticker.history(start=start, interval=interval)
            temp_hist.reset_index(inplace=True)
            temp_hist.dropna(axis=0, inplace=True)
            temp_hist.to_csv(data_path + prefix + ticker_label, index=False)
            sleep(.25)
            
    return None


def build_1mo(ticker, data_path='data/'):
    if '.csv' in ticker:
        ticker_df = pd.read_csv(data_path + '1d/' + ticker)
    else:
        ticker_df = pd.read_csv(data_path + '1d/' + ticker + '.csv')
        
    date_format = '%Y-%m-%d'
    
    # These are hardcoded and should be more changed if more flexibility is desired
    data_start = '2000-01-01'
    data_end = '2020-05-01'
    
    month_list = list(pd.date_range(data_start, data_end, freq='MS').strftime(date_format))
    
    # Not currently coded for Dividends or Stock Splits
    months_dict = {
        'Date' : [], 
        'Open' : [], 
        'High' : [], 
        'Low' : [], 
        'Close' : [], 
        'Volume' : []
    }
    
    for start in month_list:
        end = datetime.strptime(start, date_format) + relativedelta(months=1) - relativedelta(days=1)
        end = datetime.strftime(end, date_format)
        
        month_df = ticker_df.set_index('Date')[start:end].reset_index()
        
        if len(month_df) > 0:
            months_dict['Date'].append(start)
            months_dict['Open'].append(month_df.iloc[0]['Open'])
            months_dict['High'].append(max(month_df['High']))
            months_dict['Low'].append(min(month_df['Low']))
            months_dict['Close'].append(month_df.iloc[-1]['Close'])
            months_dict['Volume'].append(sum(month_df['Volume']))
    
    months_df = pd.DataFrame.from_dict(months_dict)
    return months_df


def build_data(interval, data_path='data/'):

    # interval_set = {'1mo', '5d', '1wk', '3mo'}
    interval_set = {'1mo'}
    if interval == '1d':
        return None
    if interval not in interval_set:
        print('Invalid interval')
        return -1
    
    prefix = interval + '/'
    
    os.makedirs(data_path + prefix, exist_ok=True)
    
    ticker_labels = os.listdir(data_path + '1d/')
    
    interval_function = globals()['build_' + interval]
    
    for ticker_label in ticker_labels:
        ticker_df = interval_function(ticker_label)
        ticker_df.to_csv(data_path + prefix + ticker_label, index=False)
    return None


def check_ticker(ticker, offset, interval='1mo', data_path='data/'):
    prefix = interval + '/'
    ticker_df = pd.read_csv(data_path + prefix + ticker + '.csv').set_index('Date')
    ticker_df.index = pd.to_datetime(ticker_df.index)
    if len(ticker_df) >= offset:
        return ticker_df
    return False


def build_returns(interval, data_path='data/'):
    prefix = interval + '/'
    os.makedirs(data_path + prefix, exist_ok=True)

    ticker_labels = os.listdir(data_path + prefix)

    for ticker_label in ticker_labels:
        ticker_df = pd.read_csv(data_path + prefix + ticker_label)
        ticker_df['Return'] = (ticker_df['Close'] - ticker_df['Open'])/ticker_df['Open']
        ticker_df.to_csv(data_path + prefix + ticker_label, index=False)
    return None


def fixed_long_short(returns_dict, long=15, short=0):
    allocation = {
        'long' : [], 
        'short' : []
    }
    top = sorted(returns_dict.items(), key=lambda x: x[1])[::-1]
    sorted_tickers = [x[0] for x in top]
    
    if long == 0:
        allocation['long'] == []
    else:
        allocation['long'] = sorted_tickers[:long]
    if short == 0:
        allocation['short'] = []
    else:
        allocation['short'] = sorted_tickers[-1 * short:]
    
    return allocation

class Backtest:
    # TODO Add usability to eliminate tickers at initialization (check dates)
    def __init__(self, tickers, hist_depth=None, train_depth=None, features=[], 
        data_path = 'data/', interval = '1mo', data_start = '2001-01-01', 
        start_date = '2015-01-01', end_date = '2019-12-01', 
        target='Return', download_new=False):

        self.portfolio = {}
        self.tickers = tickers
        self.features = features
        self.target = target
        
        self.hist_depth = hist_depth
        self.train_depth = train_depth
        
        self.interval = interval
        self.data_start = data_start
        
        self.data_path = data_path
        self.prefix = str(interval + '/')

        self.start_date = start_date
        self.end_date = end_date
        
        self.results = []
        self.specific_returns = []
        
        self.blacklist = set()

        self.build_portfolio(get_new=download_new)

    
    def build_portfolio(self, offset=True, get_new=False):
        if type(offset) == bool:
            offset = self.train_depth + self.hist_depth + 60 + 12 
        
        get_data(
            self.tickers, data_path=self.data_path, 
            start=self.data_start, get_new=get_new
            )
        
        ticker_dict = {}
        
        for ticker in self.tickers:
            ticker_df = check_ticker(
                ticker, offset, interval=self.interval, 
                data_path=self.data_path
                )
            if type(ticker_df) != bool:
                if 'Date' in ticker_df:
                    ticker_df.set_index('Date', inplace=True)
                    ticker_df.index = pd.to_datetime(ticker_df.index)
                ticker_dict[ticker] = ticker_df
        
        self.tickers = list(ticker_dict.keys())
        self.portfolio = ticker_dict
        return ticker_dict


    def _check_date(self, ticker, date):
        dates = self.portfolio[ticker].index
        return date in dates
    

    def clean_portfolio(self, date):
        for ticker in self.tickers:
            if not self._check_date(ticker, date):
                self.tickers.remove(ticker)
        return None


    # def get_returns(self, symbols, date):
    #     returns = []
    #     for ticker in symbols:
    #         temp_ticker_dict = self.portfolio[ticker].loc[date]
    #         returns.append((temp_ticker_dict['Close'] - temp_ticker_dict['Open'])/temp_ticker_dict['Open'])
    #     return returns


    def build_train(self, date):
        X = np.zeros((self.train_depth * len(self.portfolio), self.hist_depth * len(self.features)))
        y = np.zeros(self.train_depth * len(self.portfolio))
        j = 0
        for ticker in self.tickers:
            ticker_df = self.portfolio[ticker]
            date_i = ticker_df.index.get_loc(date)
            for i in range(1, self.train_depth + 1):
                start = date_i - i - self.hist_depth
                end = date_i - i 
                X[j] = ticker_df.iloc[start:end][self.features].values.flatten()
                y[j] = ticker_df.iloc[date_i - i + 1][self.target]
                j += 1
        return X, y
    
    
    def build_test(self, date):
        X = np.zeros((len(self.portfolio), self.hist_depth * len(self.features)))
        j = 0
        tickers = []
        for ticker in self.tickers:
            ticker_df = self.portfolio[ticker]
            date_i = ticker_df.index.get_loc(date)
            start = date_i - self.hist_depth
            end = date_i
            X[j] = ticker_df.iloc[start:end][self.features].values.flatten()
            j += 1
            tickers.append(ticker)
        return X, tickers
    

    def build_machine(self, model, date):
        X, y = self.build_train(date)

        if 'skorch' in str(type(model)):
            X = X.astype('float32')
            y = y.astype('float32')
            
            y = y.reshape(-1, 1)

        scaler_X = StandardScaler()
        scaler_X.fit(X)
        X = scaler_X.transform(X)
        
        model.fit(X, y)

        test_df, tickers = self.build_test(date)
        X_test = test_df

        if 'skorch' in str(type(model)):
            X_test = X_test.astype('float32')

        X_test = scaler_X.transform(X_test)

        predicted_returns = model.predict(X_test)
        predicted_returns = np.ravel(predicted_returns)
        predicted_returns = list(predicted_returns)
    
        returns_dict = {}
        
        for i in range(len(tickers)):
            returns_dict[tickers[i]] = predicted_returns[i]
        
        return returns_dict
        

    def backtest(self, model, allocation_builder=fixed_long_short, alloc_params = {'long': 15, 'short': 0}):
        months = list(pd.date_range(self.start_date, self.end_date, freq='MS').strftime('%Y-%m-%d'))

        for ticker in self.blacklist:
            self.tickers.remove(ticker)

        overall_returns = []
        specific_returns = []

        for month in months:
            start_time = time()

            self.clean_portfolio(month)

            returns_dict = self.build_machine(model, month)

            allocation = allocation_builder(returns_dict, **alloc_params)

            long_returns = [
                self.portfolio[ticker]['Return'][month] 
                for ticker in allocation['long']
                ]
            short_returns = [
                self.portfolio[ticker]['Return'][month] * -1
                for ticker in allocation['short']
                ]
            
            total_returns = long_returns + short_returns
            average_returns = sum(total_returns)/len(total_returns)

            specific_returns_dict = {'long': {}, 'short': {}}
            for i in range(len(allocation['long'])):
                specific_returns_dict['long'][allocation['long'][i]] = long_returns[i]

            for i in range(len(allocation['short'])):
                specific_returns_dict['short'][allocation['short'][i]] = short_returns[i]
                
            specific_returns.append(specific_returns_dict)
            overall_returns.append(average_returns)

            print(month, round(average_returns, 5), round(time() - start_time, 2))

        return overall_returns, specific_returns





    # TODO Look into cleaning portfolio month by month

    # TODO Documentation and analysis functions
