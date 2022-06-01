# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 21:25:15 2021

@author: Bogdan Tudose
"""
#%% Import Packages
import pandas as pd
import base64
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import requests 
import statsmodels.api as sm #Linear regression
from datetime import datetime
from time import mktime
from bs4 import BeautifulSoup
import re  #regular expressions
from io import StringIO, BytesIO
from urllib.request import Request, urlopen  
import json

st.set_page_config(layout="wide",page_title='Stock Beta App')
#%% Import Files
@st.cache
def grabDF(fileName):
    df = pd.read_csv("StockData/" + fileName, parse_dates=['Date'],index_col=['Date'])
    df['Returns'] = df['Close'].pct_change()
    return df


def displ_pdf(pdf_file):
    # base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
    with open(pdf_file,"rb") as f: 
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="900" height="1000" type="application/pdf">' 
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="900" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# @st.cache(suppress_st_warning=True)
def displ_pdf_link(pdf_file):
    pdf_display = f'<iframe src="{pdf_file}" width="900" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

#%% Functions for timestamps    
@st.cache(allow_output_mutation=True)
def currentTime():
    return []

def updateDate():
    if len(currentTime())>0:
        currentTime().pop(0)
        currentTime().append(datetime.now())
    else:
        currentTime().append(datetime.now())


#%% Yahoo Finance Functions
#Source: https://maikros.github.io/yahoo-finance-python/

def get_crumbs_and_cookies(stock):
    """
    get crumb and cookies for historical data csv download from yahoo finance
    parameters: stock - short-handle identifier of the company 
    returns a tuple of header, crumb and cookie
    """
    
    url = 'https://finance.yahoo.com/quote/{}/history'.format(stock)
    with requests.session():
        header = {'Connection': 'keep-alive',
                   'Expires': '-1',
                   'Upgrade-Insecure-Requests': '1',
                   'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) \
                   AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36'
                   }
        
        website = requests.get(url, headers=header)
        # soup = BeautifulSoup(website.text, 'lxml')
        soup = BeautifulSoup(website.text)
        crumb = re.findall('"CrumbStore":{"crumb":"(.+?)"}', str(soup))

        return (header, crumb[0], website.cookies)
    
def convert_to_unix(date):
    """
    converts date to unix timestamp
    parameters: date - in format (dd-mm-yyyy)
    returns integer unix timestamp
    """
    datum = datetime.strptime(date, '%d-%m-%Y')
    
    return int(mktime(datum.timetuple())) + 86400 #adding 1 day due to timezone issue


# @st.cache
def fnYFinHist(stock, interval='1d', day_begin='01-01-2013', day_end='17-11-2021'):
    """
    queries yahoo finance api to receive historical data in csv file format
    
    parameters: 
        stock - short-handle identifier of the company
        interval - 1d, 1wk, 1mo - daily, weekly monthly data
        day_begin - starting date for the historical data (format: dd-mm-yyyy)
        day_end - final date of the data (format: dd-mm-yyyy)
    
    returns a list of comma seperated value lines
    """
    
    day_begin_unix = convert_to_unix(day_begin)
    day_end_unix = convert_to_unix(day_end)
    header, crumb, cookies = get_crumbs_and_cookies(stock)
    
    with requests.session():
        url = 'https://query1.finance.yahoo.com/v7/finance/download/' \
              '{stock}?period1={day_begin}&period2={day_end}&interval={interval}&events=history&crumb={crumb}' \
              .format(stock=stock, 
                      day_begin=day_begin_unix, day_end=day_end_unix,
                      interval=interval, crumb=crumb)
                
        website = requests.get(url, headers=header, cookies=cookies)

    data = pd.read_csv(StringIO(website.text), parse_dates=['Date'], index_col=['Date'])
    data['Returns'] = data['Close'].pct_change()
    return data

# @st.cache
def fnYFinJSON(stock, field):
    if not stock:
        return "enter a ticker"
    else:
    	urlData = "https://query2.finance.yahoo.com/v7/finance/quote?symbols="+stock
    	webUrl = urlopen(urlData)
    	if (webUrl.getcode() == 200):
    		data = webUrl.read()
    	else:
    	    print ("Received an error from server, cannot retrieve results " + str(webUrl.getcode()))
    	yFinJSON = json.loads(data)
        
    try:
        tickerData = yFinJSON["quoteResponse"]["result"][0]
    except:
        return "N/A"
    if field in tickerData:
        return tickerData[field]
    else:
        return "N/A"

#%% Refresh Pricing Functions    
@st.cache
def grabPricing(ticker, field):
    fieldValue = fnYFinJSON(ticker, field)
    updateDate()
    return fieldValue

@st.cache
def grabPricingAll(ticker, interval, start, end):
    df = fnYFinHist(ticker, interval, start, end)
    updateDate()
    return df


@st.cache(allow_output_mutation=True)
def refreshPricing(ticker, timeStamp):  
    newPrice = fnYFinJSON(ticker, 'regularMarketPrice')
    return newPrice

@st.cache(allow_output_mutation=True)
def refreshPricingAll(ticker, interval, start, end, timeStamp):  
    df = fnYFinHist(ticker, interval, start, end)
    return df

#%% Streamlit Components
stocks = ['AAPL','BA','JNJ','KO','MCD','NKE',
          'ULVR.L','BP.L','RIO.L','BATS.L',
          'CGX.TO','SHOP.TO','RY.TO','BMO.TO','BNS.TO','TD.TO','CM.TO','ENB.TO',
          'NEM','GOLD','KGC',
          'TM','7203.T','SONY','6758.T','SFTBY','9984.T']
indices = ['S&P 500','Russell 2000','FTSE 100','Nikkei 225','Gold','S&P/TSX','S&P/ASX']
intervalsMap = {'Daily':'1d','Weekly':'1wk','Monthly':'1mo'}

st.sidebar.header("Model Assumptions")
ownTicker = st.sidebar.checkbox("Enter your own ticker")
with st.sidebar.form(key='inputs_form'):
    
    # if st.checkbox("Enter your own ticker") or ownTicker:
    if ownTicker:
        dropValue = "AAPL"
        stockDrop = st.text_input('Stock Ticker', placeholder=dropValue)
        if stockDrop == "":        
            stockDrop = dropValue
    else:
        stockDrop = st.selectbox('Stock Ticker:',stocks)

    indexDrop = st.selectbox('Market Index:',indices)
    intervalDrop = st.selectbox('Interval:',intervalsMap.keys(),index=2)
    interval = intervalsMap[intervalDrop]
    startDate = st.date_input('Start Date', pd.to_datetime('2016-11-01'))
    endDate = st.date_input('End Date', datetime.now())
    addConst = st.checkbox('Add Constant', value=True)
    submit_btn = st.form_submit_button(label='submit')

#Old demo codes without the YFin scrapes
# stockDF = grabDF(stockDrop + ".csv")
# indexDF = grabDF(indexDrop + ".csv")
#%% Grab Data
indexTickersMap = {'S&P 500':"^GSPC",'Russell 2000':'^RUT','FTSE 100':'^FTSE',
                   'Nikkei 225':'^N225','Gold':'GC=F','S&P/TSX':'^GSPTSE','S&P/ASX':'STW.AX'}
indexTicker = indexTickersMap[indexDrop]

#dates formatted for the YFin API
dayStart = '{:%d-%m-%Y}'.format(startDate)
dayEnd = '{:%d-%m-%Y}'.format(endDate)

stockDF = grabPricingAll(stockDrop, interval, dayStart, dayEnd)
indexDF = grabPricingAll(indexTicker, interval, dayStart, dayEnd)

stockName = grabPricing(stockDrop, 'displayName')
if stockName == 'N/A':
    stockName = grabPricing(stockDrop, 'shortName')
    
indexName = grabPricing(indexTicker , 'shortName')
stockPrice = grabPricing(stockDrop, 'regularMarketPrice')
indexPrice = grabPricing(indexTicker, 'regularMarketPrice')
stockPriceChg = grabPricing(stockDrop, 'regularMarketChange')
indexPriceChg = grabPricing(indexTicker, 'regularMarketChange')
stockPctChg = grabPricing(stockDrop, 'regularMarketChangePercent')
indexPctChg = grabPricing(indexTicker, 'regularMarketChangePercent')
stockCurrency = grabPricing(stockDrop,'currency')

currencyMap = {'GBp':'GBp','USD':'US$','CAD':'C$','JPY':'¥'}
currency = currencyMap[stockCurrency]

if st.sidebar.button("Refresh Pricing"):
    updateDate()
    indexPrice = refreshPricing(indexTicker, currentTime()[0])
    stockPrice = refreshPricing(stockDrop, currentTime()[0])
    stockDF = refreshPricingAll(stockDrop, interval, dayStart, dayEnd, currentTime()[0])
    indexDF = refreshPricingAll(indexTicker, interval, dayStart, dayEnd, currentTime()[0])
    updateDate()
    st.sidebar.write("Last price update: {}".format(currentTime()[0]))
else:
    st.sidebar.write("Last price update: {}".format(currentTime()[0]))

#%% Merging Data Sets
mergedData = indexDF.merge(stockDF, how='inner',
                         left_index=True, right_index=True,
                         suffixes=("_Index","_Stock"))  
mergedData.dropna(inplace=True)
mergedData['Date'] = mergedData.index
    #same as: mergedData = mergedData.dropna()
#%% Regression Model
mergedData['Constant'] = 1 #used to calculate alpha or y-intercept 
if addConst:
    capm = sm.OLS(mergedData['Returns_Stock'], mergedData[['Returns_Index','Constant']])
else:    
    capm = sm.OLS(mergedData['Returns_Stock'], mergedData['Returns_Index'])
results = capm.fit()
summary = results.summary() 

if addConst:
    mergedData['Predictions'] = results.predict(mergedData[['Returns_Index','Constant']])
else:
    mergedData['Predictions'] = results.predict(mergedData['Returns_Index'])
r2 = results.rsquared
beta = results.params[0]
#%% Visualizations
chartTitle = "Linear Regression {} vs {}".format(stockDrop, indexDrop)
subTitle = "R2:{:.4f} Beta:{:.4f}".format(r2, beta)
fullTitle = "{} <br><sup>{}</sup>".format(chartTitle, subTitle)
xAxisTitle = "{} {} Returns".format(indexDrop, intervalDrop)
yAxisTitle = "{} {} Returns".format(stockDrop, intervalDrop)

#Matplotlib Visualization
fig, ax = plt.subplots()
ax.plot(mergedData['Returns_Index'],mergedData['Predictions'],'red')
ax.scatter(mergedData['Returns_Index'],mergedData['Returns_Stock'],alpha=0.5)
fig.suptitle(chartTitle)
ax.set_xlabel(xAxisTitle)
ax.set_ylabel(yAxisTitle)

#Plotly Visualizations
figIndex = px.line(indexDF, x=indexDF.index, y='Close', title="{} - {}".format(indexTicker, indexName))
figStock = px.line(stockDF, x=stockDF.index, y='Close', title="{} - {}".format(stockDrop, stockName))
figIndexCandle = go.Figure(data=[go.Candlestick(x=indexDF.index,
                open=indexDF['Open'], high=indexDF['High'],
                low=indexDF['Low'], close=indexDF['Close'])
                     ])
figIndexCandle.update_layout(xaxis_rangeslider_visible=False)
figStockCandle = go.Figure(data=[go.Candlestick(x=stockDF.index,
                open=stockDF['Open'], high=stockDF['High'],
                low=stockDF['Low'], close=stockDF['Close'])
                     ])
figStockCandle.update_layout(xaxis_rangeslider_visible=False)

figRegr = px.scatter(mergedData, x='Returns_Index', y='Returns_Stock', trendline='ols')
fig1 = px.line(mergedData, x="Returns_Index", y="Predictions")
fig1.update_traces(line=dict(color = 'red'))

fig2 = px.scatter(mergedData, x="Returns_Index", y="Returns_Stock", opacity=0.5,
                  labels={'Returns_Index':indexName,'Returns_Stock':stockName},
                  hover_data={'Returns_Index':':.2%',
                              'Returns_Stock':':.2%',
                              'Date':'|%d-%b-%Y'})
fig2.update_traces(marker={'size':10})
fig3 = go.Figure(data=fig1.data + fig2.data,
                 layout={'title':fullTitle,
                         'xaxis':{'title':xAxisTitle},
                         'yaxis':{'title':yAxisTitle}})
#%% Streamlit Outputs
st.title("Demo - Stock Beta Calculator")

#KPIs
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(stockName, stockDrop)
col2.metric("Index", indexDrop)
col3.metric("R2","{:.4f}".format(r2))
col4.metric("Beta","{:.4f}".format(beta))
col5.metric("Beta 95% Conf Interval","{:.4f} - {:.4f}".format(results.conf_int().iloc[0,0],results.conf_int().iloc[0,1]))

st.metric("{} returns".format(intervalDrop),
          "{:%d-%b-%Y} to {:%d-%b-%Y}".format(startDate,endDate))

#PDF Explaining the concepts
st.header("Background on Beta and Statsmodels OLS")
st.write("Helpful links:")
aLinks = '''
<a href="https://finance.yahoo.com/news/beta-everything-know-measuring-stock-170027831.html" target="_blank">What is Beta - Yahoo Finance</a><br>
<a href="https://www.statsmodels.org/dev/examples/notebooks/generated/ols.html" target="_blank">statsmodels - OLS</a><br>
<a href="https://marqueegroup.ca/course/python-2-visualization-and-analysis/" target="_blank">Marquee Python Course</a>

'''
st.markdown(aLinks, unsafe_allow_html=True)
if st.checkbox("Show PDF slides"):
    displ_pdf("slides-web.pdf")
    
#Current Price Charts
st.header('Market Data')
liveLinks = '''
<a href="https://finance.yahoo.com/quote/{}/key-statistics?p={}" target="_blank">{} - YFinance Profile</a><br>
'''.format(stockDrop,stockDrop, stockDrop)
st.markdown(liveLinks, unsafe_allow_html=True)
if st.checkbox("Show Summary Charts"):
    chartType = st.radio("Pick chart type",('Normal','Candlestick'))
    chart1, chart2= st.columns(2)
    with chart1:
        st.metric("{} - {}".format(stockDrop, stockName),
                  "{}{:,.2f}".format(currency,stockPrice),
                  "{:,.2f} ({:.2%})".format(stockPriceChg,stockPctChg/100))
        if chartType =='Normal':
            st.plotly_chart(figStock)
        else:
            st.plotly_chart(figStockCandle)
    with chart2:
        st.metric("{} - {}".format(indexTicker, indexName),
                  "{:,.2f}".format(indexPrice),
                  "{:,.2f} ({:.2%})".format(indexPriceChg,indexPctChg/100))
        if chartType =='Normal':
            st.plotly_chart(figIndex)
        else:
            st.plotly_chart(figIndexCandle)

#Regression Outputs
col6, col7  = st.columns([1,1])
with col6:
    st.header("Regression Output")
    st.write(summary)
with col7:
    st.header("Regression Line of Best Fit")
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>', unsafe_allow_html=True)
    layoutPick = st.radio('Graphing library',['Plotly','Matplotlib'])
    if layoutPick == 'Plotly': 
        fig3.update_layout(height=600)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.pyplot(fig)
st.header("Regression Line with Plotly")
st.plotly_chart(figRegr)

#Dataframe output    
st.header("Combined Data Set:")
st.write(mergedData)
#%% Download Data
@st.cache
def createOLSFile(modelResults, fileType):
    results_summary = modelResults.summary()
    if fileType == 'csv':
        fileData = results_summary.as_csv()
    elif fileType == 'html':
        fileData = results_summary.as_html()
    return fileData

@st.cache
def createHtml(chart1, chart2, chart3):
    string1 = chart1.to_html(full_html=False, include_plotlyjs='cdn')
    string2 = chart2.to_html(full_html=False, include_plotlyjs='cdn')
    string3 = chart3.to_html(full_html=False, include_plotlyjs='cdn')
    string4 = createOLSFile(results, 'html')
    allHtml = "".join([string1, string2, string3, string4])
    return allHtml
    

@st.cache(allow_output_mutation=True)
def createExcel():
    buffer = BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        stockDF.to_excel(writer, sheet_name=stockDrop, index=True)
        indexDF.to_excel(writer, sheet_name=indexDrop, index=True)
        mergedData.to_excel(writer, sheet_name="Merged", index=True)
    return buffer

st.sidebar.header("Download Outputs")
st.sidebar.download_button(
    label="Download Plotly Graphs + OLS",
    data=createHtml(fig3,figStock,figIndex),
    file_name='regression.html')        

st.sidebar.download_button(
    label="Download data as Excel",
    data=createExcel(),
    file_name='stock_prices.xlsx')        
        
#Download OLS Results
olsFileType = st.sidebar.radio('OLS Export Format',['csv','html'])
fileName = 'ols.csv' if olsFileType == 'csv' else 'ols.html'

st.sidebar.download_button(
    label="Download OLS results",
    data = createOLSFile(results, olsFileType), 
    file_name = fileName) 
#%% TESTS
st.header("Marquee Course Outlines")
st.write("Marquee Python course links:")
courseWebLinks = '''
<a href="https://marqueegroup.ca/course/python-1-core-data-analysis/" target="_blank">Python 1: Core Data Analysis</a><br>
<a href="https://marqueegroup.ca/course/python-2-visualization-and-analysis/" target="_blank">Python 2: Visualization and Analysis</a><br>
<a href="https://marqueegroup.ca/course/python-3-web-scraping-and-machine-learning/" target="_blank">Python3: Web Scraping and Machine Learning</a>
'''
st.markdown(courseWebLinks, unsafe_allow_html=True)

st.markdown("*Note the embedded pdfs below might be blocked in Chrome depending on your security settings. If so, try a different browser (Firefox or Safari).*")
course = st.radio('Pick a course outline',['Python 1','Python 2','Python 3'])
# pdf_file = st.file_uploader("Marquee Course Outlines", type=["pdf"])
# if pdf_file is not None:
#     save_image_path = './StockData/'+pdf_file.name
#     with open(save_image_path, "wb") as f:
#         f.write(pdf_file.getbuffer())
#     displ_pdf(save_image_path)
    
#public link


pdfLinks = {"Python 1": "https://marqueegroup.ca/wp-content/uploads/2020/11/Python-1-Core-Data-Analysis-Outline.pdf",
            "Python 2": "https://marqueegroup.ca/wp-content/uploads/2020/11/Python-2-Visualization-and-Analysis-Outline.pdf",
            "Python 3": "https://marqueegroup.ca/wp-content/uploads/2020/11/Python-3-Web-Scraping-and-Machine-Learning-Outline.pdf"
    }
pdfLink = pdfLinks[course]
# pdf_display = F'<iframe src="{pdfLink}" width="900" height="1000" type="application/pdf"></iframe>'
# pdf_display = f'<embed src="{pdfLink}" width="900" height="1000" type="application/pdf">' 

# st.markdown(pdf_display, unsafe_allow_html=True)
displ_pdf_link(pdfLink)