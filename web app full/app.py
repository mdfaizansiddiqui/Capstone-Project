from flask import Flask,render_template,request,redirect,send_file,url_for
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from RLBot import trade
import matplotlib.pyplot as plt
import sys

import chart_studio
import chart_studio.plotly as py
import chart_studio.tools as tls
import plotly.express as px
#put the folder path of the model 
#sys.path.insert(1,'/Users/FAIZAN/Desktop/capstone web app/capstone phase-2 new work/capstone phase-2 new work')
#Multi_company_RL=__import__('Multi_company RL')
import pickle
import numpy as np
import base64

app = Flask(__name__)

#def get_key(val,dictio):
#    for key, value in dictio.items():
#         if val == value:
#             return key

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST','GET'])
def predict():
    
    companies=[]
    tickers=[]
    print(request.form)
    amount = int(request.form['amount'])
    selected_company = request.form['companies']
    start_date = str(request.form['start_date'])
    end_date = str(request.form['end_date'])
    
    print(start_date)
    print(end_date)
#    start_date = request.form['start_date']
#    end_date = request.form['end_date']

    if(selected_company != "All"):
        companies.append(selected_company)
        if(selected_company=="amazon"):
            tickers.append("AMZN")
        elif(selected_company=="apple"):
            tickers.append("AAPL")
        elif(selected_company=="google"):
            tickers.append("GOOGL")
        else:
            tickers.append("MSFT")
    else:
        companies = ["amazon","apple","google","microsoft"]
        tickers = ["AMZN","AAPL","GOOGL","MSFT"]
        
    
    ret=trade(start_date, end_date,amount,companies,tickers)#list
    
    l=len(ret)
    last=ret[-1]
    roi=ret[-1]/amount
    
    username='faizans'
    api_key='AiggkGDZCUIQG7B2oidF'
    chart_studio.tools.set_credentials_file(username=username,api_key=api_key)
    
    fig1 = px.line(ret,title = 'Returns after trading')
    fig1.update_xaxes(title_text='Days')
    fig1.update_yaxes(title_text='Amount')
    plot1=py.plot(fig1,filename="Returns after trading", auto_open=False,config={'displayModeBar':False})
    print(plot1) 
    
    
    Html_file = open("templates/output.html","w")
    html_str='''
        <html>
            <head>
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
                <style>
                    body{
                        margin:0 100; background:whitesmoke;
                    }
                    .modebar-container{
                        display:none !important;
                    }
                    
                    .modebar .modebar--hover{
                    display:none !important;
                    }
                
                
                </style>
            </head>
            <body>
                
                <iframe width="1000" height="550" frameborder="0" seamless="seamless" scrolling="no" \
                src="''' + plot1 + '''.embed?width=800&height=550"></iframe><br />
                
                <div >After trading for {{l}} on an initial investment of {{amount}}</div><br>
                <div>The final return Amount is {{last}}</div><br>
                <div>Return on Investment {{roi}}</div
                
            </body>
        </html>
    '''
    
    Html_file.write(html_str)
    Html_file.close()
    
    return render_template('output.html',l=l,last=last,roi=roi,amount=amount)



if __name__ == '__main__':
    app.run(debug=True)