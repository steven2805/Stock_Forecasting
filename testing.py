from flask import Flask
import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame 

app = Flask(__name__)
data = pd.read_csv("RMG.L.csv")
data.head()
