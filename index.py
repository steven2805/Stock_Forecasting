import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame 
from flask import Flask

data = pd.read_csv("RMG.L.csv")
data.head()
