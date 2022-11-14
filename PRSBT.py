from calendar import month
from cgitb import html
from contextlib import nullcontext
from ctypes import alignment
import json
import statistics
from textwrap import fill
import time
from telnetlib import STATUS
from tkinter import scrolledtext
from turtle import bgcolor, width
import requests
from tkinter import *
from tkinter import ttk
import numpy as np
from config import client_id
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta
import math
from lightgbm import LGBMRegressor
import optuna
from sklearn.model_selection import train_test_split
from backtesting.lib import crossover
from backtesting.test import SMA
from backtesting import Backtest, Strategy
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from statistics import mean

def makeData(Event = None):
    
    psymbol.delete(0,END)
    badSym.delete(0.0,END)
    view.delete()
    view2.delete()
    if len(port) == 0:
        badSym.insert(0.0,'Portfolio Empty')
        for i in range(4,15):
            portBox3 = Text(if1, width=10, height=1, bg="#2E3138", fg="white",font=("montserrat", 8), bd=0, highlightthickness=0)
            portBox3.grid(row=3, column=i, padx=(5,5), pady=(0,0))
    else:
        for item in view.get_children():
                    view.delete(item)
        for item in view2.get_children():
                    view2.delete(item)
        for symbols in port:    #np.sort(port) for alphabetical order
            quotes = r'https://api.tdameritrade.com/v1/marketdata/{}/quotes'.format(symbols)
            quotespayload = {'apikey':client_id}
            quotescontent = requests.get(url = quotes, params = quotespayload)
            quotesjson_data = quotescontent.json()
            quotesdf = pd.DataFrame(quotesjson_data)

            if quotesdf.empty:
                badSym.delete(0.0,END)
                badSym.insert(0.0,'Symbol Not Found')
                port.remove(symbols)
            else:
                badSym.delete(0.0,END)
                portfolio[symbols] = quotesdf
                pf = portfolio.T
                pf['predictions'] = " "
                pf['rank'] = " "
                cols = pf[['rank','symbol','predictions']]

                stats = []

                for j in cols.loc[symbols]:
                    stats.append(j)
                view.insert(parent='',index='end',text='',values=stats,tags=('view'))
                view2.insert(parent='',index='end',text='',values=stats,tags=('view'))

        view.tag_configure('view',background='#2E3138',foreground='white')
        view.grid()
        view.config(yscrollcommand=scroll.set)
        scroll.config(command=view.yview)

        view2.tag_configure('view',background='#2E3138',foreground='white')
        view2.grid()
        view2.config(yscrollcommand=scroll2.set)
        scroll2.config(command=view2.yview)


def getWatchlist():
        for symbols in np.sort(watchlist):
            watchlistquotes = r'https://api.tdameritrade.com/v1/marketdata/{}/quotes'.format(symbols)
            watchlistquotespayload = {'apikey':client_id}
            watchlistquotescontent = requests.get(url = watchlistquotes, params = watchlistquotespayload)
            watchlistquotesjson_data = watchlistquotescontent.json()
            watchlistquotesdf = pd.DataFrame(watchlistquotesjson_data)

            watchlistportfolio[symbols] = watchlistquotesdf
            watchlistpf = watchlistportfolio.T
            cols = watchlistpf[['symbol','description','markPercentChangeInDouble','lastPrice']]

            stats = []

            for j in cols.loc[symbols]:
                stats.append(j)
            watchlistview.insert(parent='',index='end',text='',values=stats,tags=('watchlistview'))
        watchlistview.tag_configure('view',background='#2E3138',foreground='white')
        watchlistview.grid()
        watchlistscroll = ttk.Scrollbar(if1,orient="vertical", command = watchlistview.yview)
        watchlistscroll.grid(row=2, column=4, sticky=NS+EW, padx=(0,10))
        watchlistview.config(yscrollcommand=watchlistscroll.set)
        watchlistscroll.config(command=watchlistview.yview)
        

def addToRanking(Event=None):
    badSym.delete(0.0,END)
    symbol = psymbol.get()
    symbol = str.upper(symbol)
    if port.__contains__(symbol) == False:
        port.append(symbol)
        makeData()
    elif port.__contains__(symbol) == True:
        badSym.delete(0.0,END)
        badSym.insert(0.0,'Symbol Already In Portfolio')
    psymbol.delete(0,END)

def removeFromRanking(Event=None):
    badSym.delete(0.0,END)
    symbol = psymbol.get()
    symbol = str.upper(symbol)
    if port.__contains__(symbol) == True:
        port.remove(symbol)
        makeData()
    elif port.__contains__(symbol) == False:
        badSym.delete(0.0,END)
        badSym.insert(0.0,'Symbol Not In Portfolio')
    psymbol.delete(0,END)

def rankRanking(Event=None):
    portfolio = pd.DataFrame()
    portfolio['Symbol'] = port
    data = []
    view.delete()
    view2.delete()
    for symbols in port:
        ctime = time.time()
        ctime = (ctime.__round__()*1000)
        dtime = (ctime - 31556926000)
        phistory = r'https://api.tdameritrade.com/v1/marketdata/{}/pricehistory'.format(symbols)
        phistorypayload = {'apikey':client_id,
                        'periodType':'year',
                        'frequencyType':'daily',
                        'frequency':'1',
                        'period':'1',
                        'endDate':ctime,
                        'startDate':dtime,
                        'needExtendeedHoursData':'true'}
        phistorycontent = requests.get(url = phistory, params = phistorypayload)
        pcontent = phistorycontent.json()
        pcontent = DataFrame(pcontent['candles'])
        pcontent = pcontent.T.drop(['datetime']).T
        pcontent['symbol'] = symbols
        pcontent['range'] = (pcontent['high'] - pcontent['low']) / pcontent['close'] * 100.0
        pcontent['change'] = (pcontent['close'] - pcontent['open']) / pcontent['open'] * 100.0
        
        factors = pcontent[['open','high','low','close','volume','range','change']]
        factors = factors.dropna()
        forecast_col = 'close'
        forecast_out = int(math.ceil(0.01 * len(factors)))
        factors['forecast'] = factors[forecast_col].shift(-forecast_out)
        factors.dropna(inplace=True)

        X = factors.iloc[:,:-1]
        y = factors.forecast

        def objective(trial,data=data):
        
            X_train,X_test,y_train,y_test = train_test_split(X,y)

            params = {
            'metric': 'rmse', 
            'random_state': 48,
            'n_estimators': 20000,
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
            'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.006,0.008,0.01,0.014,0.017,0.02]),
            'max_depth': trial.suggest_categorical('max_depth', [10,20,100]),
            'num_leaves' : trial.suggest_int('num_leaves', 7, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
            'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100)
            }

            model = LGBMRegressor(**params)
            model.fit(X_train,y_train)
            prediction = model.predict(X_test)
            pred = (mean_squared_error(prediction,y_test))
            return pred
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=2)
        pred = study.best_value
    
        denom = factors['close']
        count = len(denom)-1
        denom = denom[count]
        ranks = pred/denom
        data.append([ranks])

    portfolio['Prediction'] = data
    portfolio = portfolio.sort_values(by = "Prediction", ascending=False)
    #portfolio = portfolio.drop(columns=['Prediction'])7
    portfolio = portfolio.reset_index()
    portfolio = portfolio.drop(columns=['index'])
    portfolio['Rank'] = portfolio.index+1
    indexcol = 0
    for item in view.get_children():
        view.delete(item)
    for item in view2.get_children():
        view2.delete(item)
    for symbol in portfolio['Symbol']:
        #if port.__contains__(symbol):

        quotes = r'https://api.tdameritrade.com/v1/marketdata/{}/quotes'.format(symbol)
        quotespayload = {'apikey':client_id}
        quotescontent = requests.get(url = quotes, params = quotespayload)
        quotesjson_data = quotescontent.json()
        quotesdf = pd.DataFrame(quotesjson_data)

        pf = quotesdf.T
        pf['rank'] = portfolio.at[indexcol,'Rank']
        pf['prediction'] = portfolio.at[indexcol,'Prediction']
        cols = pf[['rank','symbol','prediction']]
        rstats = []
        for j in cols.loc[symbol]:
            rstats.append(j)
        view.insert(parent='',index='end',text='',values=rstats,tags=('view'))
        view2.insert(parent='',index='end',text='',values=rstats,tags=('view'))
        indexcol = indexcol + 1
    view.tag_configure('view',background='#2E3138',foreground='white')
    view.grid()
    scroll = ttk.Scrollbar(if1,orient="vertical", command = view.yview)
    scroll.grid(row=2, column=9, sticky=NS+W)
    view.config(yscrollcommand=scroll.set)
    scroll.config(command=view.yview)

    view2.tag_configure('view',background='#2E3138',foreground='white')
    view2.grid()
    scroll2 = ttk.Scrollbar(if2,orient="vertical", command = view2.yview)
    scroll2.grid(row=1, column=4, sticky=NS+W, pady=(10,10))
    view2.config(yscrollcommand=scroll.set)
    scroll2.config(command=view2.yview)
            


def backtest(Event=None):
    symbol2 = btsymbol.get()
    symbol2 = str.upper(symbol2)
    sday = sdentry.get()
    stime = stentry.get()
    eday = edentry.get()
    etime = etentry.get()
    start_balance = int(sbentry.get())
    f"{start_balance:_d}"
    ctime = time.time()
    ctime = (ctime.__round__()*1000)
    dtime = (ctime - 31556926000)
    phistory = r'https://api.tdameritrade.com/v1/marketdata/{}/pricehistory'.format(symbol2)
    phistorypayload2 = {'apikey':client_id,
                    'periodType':'day',
                    'frequencyType':'minute',
                    'frequency':'30',
                    'period':'1',
                    'endDate':ctime,
                    'startDate':dtime,
                    'needExtendeedHoursData':'true'}
    phistorycontent2 = requests.get(url = phistory, params = phistorypayload2)
    pcontent2 = phistorycontent2.json()
    pcontent2 = DataFrame(pcontent2['candles'])
    pcontent2 = pcontent2.rename(columns={'open': 'Open', 'high': 'High', 'low':'Low', 'close':'Close', 'volume':'Volume'})
    pcontent2.index = pd.to_datetime(pcontent2['datetime'])
    #pcontent2.drop("datetime", axis=1, inplace=True)
    class SmaCross(Strategy):

        def init(self):
            close = self.data.Close
            open = self.data.Open
            self.line2 = close
            self.line1 = ((close+open)/2)

        def next(self):
            if self.line1 > self.line2:
                self.buy()
            elif self.line1 < self.line2:
                self.sell()
    bt = Backtest(pcontent2, SmaCross, cash = start_balance)
    stats = bt.run()
    #statistics = pd.DataFrame(stats)
    #nstatistics = statistics.T
    bt.plot()
    bcol = 1
    results = []
    counttttt = 0
    while counttttt < 26:
        results.append(stats[counttttt])
        counttttt = counttttt + 1
    backtestview.insert(parent='',index='end',text='',values=results,tags=('view'))
    backtestview.tag_configure('view',background='#2E3138',foreground='white')
    backtestview.grid()
    backtestscroll = ttk.Scrollbar(if2,orient="vertical", command = backtestview.yview)
    backtestscroll.grid(row=5, column=21, sticky=NS+W)
    backtestview.config(yscrollcommand=backtestscroll.set)
    backtestscroll.config(command=backtestview.yview)



interface = Tk()
interface.title("Interface")
interface.state('zoomed')
interface.iconbitmap('blacklogo.ico')
interface.configure(background='#2E3138')
interface.bind('<Return>', addToRanking)

nb = ttk.Notebook(interface, height=1920, width=1920)
nb.grid()

if1 = Frame(nb, bg = '#2E3138', border=0, background='#2E3138', )
if2 = Frame(nb, bg = '#2E3138', border=0, background='#2E3138', )

if1.bind('<Return>', addToRanking)
if2.bind('<Return>', backtest)

if1.grid(row=0, column=3, sticky='W', padx=(0,0), pady=(0,0))
if2.grid(row=0, column=2, sticky='W', padx=(0,0), pady=(0,0))

nb.add(if1,text='Portfolio')
nb.add(if2,text='Backtesting')

#  communicate back to the scrollbar

Label(if2, text='Symbol',bg="#2E3138", fg="white", font=("montserrat", 8), bd=0, highlightthickness=0).grid(row=1, column=5,sticky='W', padx=(0,0), pady=(10,10))
btsymbol = Entry(if2, width=8, bg="#23252B", fg="white", font=("montserrat", 8), bd=0, highlightthickness=0, justify='center')
btsymbol.grid(row=1, column=6, sticky='W', padx=(0,0), pady=(10,10))


Label(if2, text='Start Date',bg="#2E3138", fg="white", font=("montserrat", 8), bd=0, highlightthickness=0).grid(row=1, column=7,sticky='W', padx=(0,0), pady=(10,10))
sdentry = Entry(if2, width=8, bg="#23252B", fg="white", font=("montserrat", 8), bd=0, highlightthickness=0, justify='center')
sdentry.insert(0,'6/1/2022')
sdentry.grid(row=1, column=8, sticky='W', padx=(0,0), pady=(10,10))

Label(if2, text='Start Time',bg="#2E3138", fg="white", font=("montserrat", 8), bd=0, highlightthickness=0).grid(row=1, column=9,sticky='W', padx=(0,0), pady=(10,10))
stentry = Entry(if2, width=8, bg="#23252B", fg="white", font=("montserrat", 8), bd=0, highlightthickness=0, justify='center')
stentry.insert(0,'9:30')
stentry.grid(row=1, column=10, sticky='W', padx=(0,0), pady=(10,10))


Label(if2, text='End Date',bg="#2E3138", fg="white", font=("montserrat", 8), bd=0, highlightthickness=0).grid(row=1, column=11,sticky='W', padx=(0,0), pady=(10,10))
edentry = Entry(if2, width=8, bg="#23252B", fg="white", font=("montserrat", 8), bd=0, highlightthickness=0, justify='center')
edentry.insert(0,'7/1/2022')
edentry.grid(row=1, column=12, sticky='W', padx=(0,0), pady=(10,10))

Label(if2, text='End Time',bg="#2E3138", fg="white", font=("montserrat", 8), bd=0, highlightthickness=0).grid(row=1, column=13,sticky='W', padx=(0,0), pady=(10,10))
etentry = Entry(if2, width=8, bg="#23252B", fg="white", font=("montserrat", 8), bd=0, highlightthickness=0, justify='center')
etentry.insert(0,'9:30')
etentry.grid(row=1, column=14, sticky='W', padx=(0,0), pady=(10,10))


Label(if2, text='Balance',bg="#2E3138", fg="white", font=("montserrat", 8), bd=0, highlightthickness=0).grid(row=1, column=15,sticky='W', padx=(0,0), pady=(10,10))
sbentry = Entry(if2, width=8, bg="#23252B", fg="white", font=("montserrat", 8), bd=0, highlightthickness=0, justify='center')
sbentry.insert(0,'1000000')
sbentry.grid(row=1, column=16, sticky='W', padx=(0,0), pady=(10,10))

Button(if2, text="Search", command=backtest, height=1, fg="#2E3138", font=("montserrat", 8), bd=0, highlightthickness=0) .grid(row=1, column=17, sticky='n', padx=(0,0), pady=(65,65))

backtestview = ttk.Treeview(if2,height=20)
backtestview.grid(row=5, column=1, padx=(0,0), pady=(0,0),rowspan=1, columnspan=20, sticky=NSEW)
backtestview['columns'] = ('Start', 'End', 'Duration', 'Exposure Time [%]', 'Equity Final [$]',
       'Equity Peak [$]', 'Return [%]', 'Buy & Hold Return [%]',
       'Return (Ann.) [%]', 'Volatility (Ann.) [%]', 'Sharpe Ratio',
       'Sortino Ratio', 'Calmar Ratio', 'Max. Drawdown [%]',
       'Avg. Drawdown [%]', 'Max. Drawdown Duration', 'Avg. Drawdown Duration',
       '# Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]',
       'Avg. Trade [%]', 'Max. Trade Duration', 'Avg. Trade Duration',
       'Profit Factor', 'Expectancy [%]')
backtestview.column("#0", width=0,  stretch=NO)
backtestview.heading("#0",text="",anchor=CENTER)
for i in backtestview['columns']:
    backtestview.column(i,anchor=CENTER, width=55)
    backtestview.heading(i,text=str(i),anchor=CENTER)
backtestscroll = ttk.Scrollbar(if2,orient="vertical", command = backtestview.yview)
backtestscroll.grid(row=5, column=21, sticky=NS+EW)
backtestview.config(yscrollcommand=backtestscroll.set)

watchlist = ['APPS','GOOG','MSFT','TSLA','FAZE','AAL','CCL','DAL','UAL','USO','BILZF','SNAP','META','GLD','SVM']
#watchlist = []
port = []
#commercial_services = pd.read_html('https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/commercial-services/')
#communications = pd.read_html('https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/communications/')
#consumer_durables = pd.read_html('https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/consumer-durables/')
#consumer_non_durables = pd.read_html('https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/consumer-non-durables/')
#consumer_services = pd.read_html('https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/consumer-services/')
#distribution_services = pd.read_html('https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/distribution-services/')
#electronic_technology = pd.read_html('https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/electronic-technology/')
#energy_minerals = pd.read_html('https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/energy-minerals/')
#finance = pd.read_html('https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/finance/')
#health_services = pd.read_html('https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/health-services/')
#health_technology = pd.read_html('https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/health-technology/')
#industrial_services = pd.read_html('https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/industrial-services/')
#miscellaneous = pd.read_html('https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/miscellaneous/')
#non_energy_minerals = pd.read_html('https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/non-energy-minerals/')
#process_industries = pd.read_html('https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/process-industries/')
#producer_manufacturing = pd.read_html('https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/producer-manufacturing/')
#retail_trade = pd.read_html('https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/retail-trade/')
#technology_services = pd.read_html('https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/technology-services/')
#transportation = pd.read_html('https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/transportation/')
#utilities = pd.read_html('https://www.tradingview.com/markets/stocks-usa/sectorandindustry-sector/utilities/')
#airlines = pd.read_html('https://www.tradingview.com/markets/stocks-usa/sectorandindustry-industry/airlines/')

#for q in commercial_services:
#    string =  (q['Unnamed: 0'])
#    allSyms = string.str.split()
#    for z in allSyms:
#        if len(z[0]) >= 2:
#            port.append(z[0])
#        elif len(z[0]) == 1 and len(z[1]) <= 4: 
#            port.append(z[1])
#        elif len(z[0]) == 1 and len(z[1]) >= 4:
#            port.append(z[0])
#for symbols in port:
#    if symbols.__contains__('.'):
#        port.remove(symbols)

portfolio = pd.DataFrame()
watchlistportfolio = pd.DataFrame()

psymbol = Entry(if1, width=10, bg="#23252B", fg="white", font=("montserrat", 15), bd=0, highlightthickness=0, justify='center')
psymbol.grid(row=1, column=6, sticky='w', padx=(0,20), pady=(0,0))

addButton = Button(if1, text="Add", command=addToRanking, height=1, fg="#2E3138", font=("montserrat", 10), bd=0, highlightthickness=0, width= 7)
addButton.grid(row=1, column=7, sticky=E, padx=(5,5), pady=(10,10))

removeButton = Button(if1, text="Remove", command=removeFromRanking, height=1, fg="#2E3138", font=("montserrat", 10), bd=0, highlightthickness=0, width= 7)
removeButton.grid(row=1, column=8, sticky=W, padx=(0,5), pady=(10,10))

rankButton = Button(if1, text="Rank", command=rankRanking, height=1, fg="#2E3138", font=("montserrat", 10), bd=0, highlightthickness=0, width= 7)
rankButton.grid(row=1, column=9, sticky=W, padx=(0,100), pady=(10,10))

badSym = Text(if1, bg="#2E3138", width = 25, height=1,fg="red", font=("montserrat", 12), bd=0, highlightthickness=0)
badSym.grid(row=3, column=1, padx=(0,0), pady=(0,0))
badSym.tag_configure("center", justify='center')
badSym.tag_add("center", 1.0, "end")

style = ttk.Style(if1)
style.theme_use("clam")
style.configure("Treeview", background="#2E3138", fieldbackground="#2E3138", foreground="white")


view = ttk.Treeview(if1,height=10)
view.grid(row=2, column=6, padx=(0,0), pady=(0,0),rowspan=1, sticky=NSEW, columnspan=3)
view['columns'] = ('Rank','Symbol','Prediction')
view.column("#0", width=0,  stretch=NO)
view.heading("#0",text="",anchor=CENTER)
for i in view['columns']:
    view.column(i,anchor=CENTER, width=80)
    view.heading(i,text=str(i),anchor=CENTER)
scroll = ttk.Scrollbar(if1,orient="vertical", command = view.yview)
scroll.grid(row=2, column=9, sticky=NS+W)
view.config(yscrollcommand=scroll.set)

view2 = ttk.Treeview(if2,height=5)
view2.grid(row=1, column=1, padx=(0,0), pady=(10,10),rowspan=1, sticky=NSEW, columnspan=3)
view2['columns'] = ('Rank','Symbol','Prediction')
view2.column("#0", width=0,  stretch=NO)
view2.heading("#0",text="",anchor=CENTER)
for i in view2['columns']:
    view2.column(i,anchor=CENTER, width=80)
    view2.heading(i,text=str(i),anchor=CENTER)
scroll2 = ttk.Scrollbar(if2,orient="vertical", command = view2.yview)
scroll2.grid(row=1, column=4, sticky=NS+W, pady=(10,10))
view2.config(yscrollcommand=scroll2.set)

wlvheight = (len(watchlist))

watchlistview = ttk.Treeview(if1,height=wlvheight)
watchlistview.grid(row=2, column=1, padx=(0,0), pady=(0,0),rowspan=1, columnspan=3, sticky=NSEW)
watchlistview['columns'] = ('Symbol','Company','Change %','Last')
watchlistview.column("#0", width=0,  stretch=NO)
watchlistview.heading("#0",text="",anchor=CENTER)
for i in watchlistview['columns']:
    watchlistview.column(i,anchor=CENTER, width=80)
    watchlistview.heading(i,text=str(i),anchor=CENTER)
watchlistscroll = ttk.Scrollbar(if1,orient="vertical", command = watchlistview.yview)
watchlistscroll.grid(row=2, column=4, sticky=NS+EW, padx=(0,10))
watchlistview.config(yscrollcommand=watchlistscroll.set)


#makeData()
getWatchlist()

interface.mainloop()