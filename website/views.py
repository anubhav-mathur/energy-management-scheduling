from django.shortcuts import render
from .models import Prediction
# from django.views import generic
# from create.models import create_code
# from django.views.generic.edit import CreateView
# from django.shortcuts import render,redirect
from django.http import HttpResponseRedirect
import pandas as pd;
import numpy as np;
import sys;
import xlsxwriter
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import LSTM

from subprocess import run,PIPE
from django.http import HttpResponse
from django.test import override_settings


from datetime import datetime, date

def index(request):
    pred= Prediction.objects.all()
    context={'pred' : pred}
    return render(request,'major/index.html',context)

def detail(request):
    pred= Prediction.objects.all()
    context={'pred' : pred}
    # df=pd.read_excel('C:/Users/aryak/Desktop/website/dataWeather.xlsx');
    #
    # dateAr = df['Date']
    # dateAr = dateAr.loc[6001:]
    # dateAr2 = pd.DataFrame(dateAr);
    # #dateAr2.to_excel(excel_writer = "intDate.xlsx", index=False);
    #
    # print(df.head())
    # df.drop(['DE wind (KW)'], axis=1, inplace=True)
    # df['Date'] = pd.to_datetime(df['Date'])
    # ind_df = df.set_index(['Date'], drop=True)
    # ind_df.head()
    # ind_df = ind_df.sort_index()
    # plt.figure(figsize=(10, 6))
    # # ind_df['DE solar (MW)'].plot();
    # # split_date = pd.Timestamp('20-11-2016  22:00:00')
    # # df =  df['DE solar (MW)']
    # train = df.loc[:6000]
    # test = df.loc[6001:]
    # inttr = train.set_index(['Date'], drop=True)
    # intte = test.set_index(['Date'], drop=True)
    # plt.figure(figsize=(10, 6))
    # ax = inttr.plot()
    # intte.plot(ax=ax)
    # plt.legend(['train', 'test']);
    #
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # train_sc = scaler.fit_transform(inttr)
    # test_sc = scaler.transform(intte)
    # # np.random.shuffle(train_sc)
    # X_train = train_sc[:-1]
    # y_train = train_sc[1:]
    # X_test = test_sc[:-1]
    # y_test = test_sc[1:]
    # X_train = np.reshape(X_train, (6000, 1, 1))
    # X_test = np.reshape(X_test, (2544, 1, 1))
    # lstm_model = Sequential()
    # lstm_model.add(LSTM(7, input_shape=(1, X_train.shape[1]), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
    # lstm_model.add(Dense(1))
    # lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    # early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    # history_lstm_model = lstm_model.fit(X_train, y_train, epochs=5, batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop])
    # y_pred_test_lstm = lstm_model.predict(X_test)
    # y_train_pred_lstm = lstm_model.predict(X_train)
    # ypf_train = scaler.inverse_transform(y_train_pred_lstm)
    # ypf_test = scaler.inverse_transform(y_pred_test_lstm)
    # ypf2 = pd.DataFrame(ypf_test);
    # ypf2.columns = ['DE solar (KW)']
    # dateAr2.drop(dateAr2.tail(1).index,inplace=True)
    # dateAr2['DE solar (KW)']=ypf2.values
    #
    #
    # dateAr2.to_excel(excel_writer = "dateSolar.xlsx", index=False);
    #
    #
    #
    # print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_lstm)))
    # print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_lstm)))
    # lstm_test_mse = lstm_model.evaluate(X_test, y_test, batch_size=1)
    # print('LSTM: %f'%lstm_test_mse)
    # plt.figure(figsize=(10, 6))
    # plt.plot(y_test, label='True')
    # plt.plot(y_pred_test_lstm, label='LSTM')
    # plt.title("LSTM's Prediction")
    # plt.xlabel('Observation')
    # plt.ylabel('DE solar (KW)')
    # plt.legend()
    # plt.show()
    # plt.savefig('solar.png')
    #
    #
    #
    # #WIND
    #
    # df=pd.read_excel('C:/Users/aryak/Desktop/website/dataWeather.xlsx');
    #
    # dateAr = df['Date']
    # dateAr = dateAr.loc[6001:]
    # dateAr2 = pd.DataFrame(dateAr);
    #
    # print(df.head())
    # df.drop(['DE solar (KW)'], axis=1, inplace=True)
    # df['Date'] = pd.to_datetime(df['Date'])
    # ind_df = df.set_index(['Date'], drop=True)
    # ind_df.head()
    # ind_df = ind_df.sort_index()
    # plt.figure(figsize=(10, 6))
    # # ind_df['DE solar (MW)'].plot();
    # # split_date = pd.Timestamp('20-11-2016  22:00:00')
    # # df =  df['DE solar (MW)']
    # train = df.loc[:6000]
    # test = df.loc[6001:]
    # inttr = train.set_index(['Date'], drop=True)
    # intte = test.set_index(['Date'], drop=True)
    # plt.figure(figsize=(10, 6))
    # ax = inttr.plot()
    # intte.plot(ax=ax)
    # plt.legend(['train', 'test']);
    #
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # train_sc = scaler.fit_transform(inttr)
    # test_sc = scaler.transform(intte)
    # # np.random.shuffle(train_sc)
    # X_train = train_sc[:-1]
    # y_train = train_sc[1:]
    # X_test = test_sc[:-1]
    # y_test = test_sc[1:]
    # X_train = np.reshape(X_train, (6000, 1, 1))
    # X_test = np.reshape(X_test, (2544, 1, 1))
    # lstm_model = Sequential()
    # lstm_model.add(LSTM(7, input_shape=(1, X_train.shape[1]), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
    # lstm_model.add(Dense(1))
    # lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    # early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    # history_lstm_model = lstm_model.fit(X_train, y_train, epochs=5, batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop])
    # y_pred_test_lstm = lstm_model.predict(X_test)
    # y_train_pred_lstm = lstm_model.predict(X_train)
    # ypf_train = scaler.inverse_transform(y_train_pred_lstm)
    # ypf_test = scaler.inverse_transform(y_pred_test_lstm)
    #
    # ypf2 = pd.DataFrame(ypf_test);
    # ypf2.columns = ['DE wind (KW)']
    # dateAr2.drop(dateAr2.tail(1).index,inplace=True)
    # dateAr2['DE wind (KW)']=ypf2.values
    #
    #
    # dateAr2.to_excel(excel_writer = "dateWind.xlsx", index=False);
    #
    #
    # cSolar = pd.read_excel("dateSolar.xlsx")
    # cWind = pd.read_excel("dateWind.xlsx")
    # inner_join_df = cSolar.merge(cWind, how="inner", on="Date")
    # inner_join_df.to_excel("predWeather.xlsx", index=False)
    #
    # print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_lstm)))
    # print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_lstm)))
    # lstm_test_mse = lstm_model.evaluate(X_test, y_test, batch_size=1)
    # print('LSTM: %f'%lstm_test_mse)
    # plt.figure(figsize=(10, 6))
    # plt.plot(y_test, label='True')
    # plt.plot(y_pred_test_lstm, label='LSTM')
    # plt.title("LSTM's Prediction")
    # plt.xlabel('Observation')
    # plt.ylabel('DE wind (KW)')
    # plt.legend()
    # plt.show()
    # plt.savefig('wind.png')



    return render(request,'major/detail.html',context)

def uploadsuccess(request):
    #pred=Prediction.objects.all()
    # solar=request.POST.get('solar')
    # wind=request.POST.get('wind')
    # data=Prediction(solar=solar,wind=wind);
    # data.save()
    #call first.py method
    # out=run([sys.executable,'C:/Users/aryak/Desktop/first.py',wind],shell=False,stdout=PIPE)
    # print(out)
    # print(coal)
    # return render(request,'major/detail.html',{'data1':out})
    # workbook = xlsxwriter.Workbook('result.xlsx')
    # worksheet = workbook.add_worksheet()
    # row = 0
    # column = 0
    #
    # import sys;
    #
    # weather=pd.read_excel('predWeather.xlsx');
    # household=pd.read_excel('dataHousehold.xlsx');
    #
    # X=household.iloc[3:];
    # df = pd.DataFrame(X);
    # df.to_excel(excel_writer = "final_household.xlsx",header=False,index=False);
    # new_household=pd.read_excel('final_household.xlsx');
    # new_household=new_household.loc[6001:]
    # new_household=new_household[['Total_1','Total_2','Total_3','Total_4','Total_5','Total_6']];
    #
    # coalFrame = [[0]*6]*2530
    # coalFrame=pd.DataFrame(coalFrame)
    # for i in range(2544):
    #     for j in range(6):
    #         if new_household.iat[i,j] < 0 :
    #             new_household.iat[i,j] = 0
    #
    # for i in range(2544):
    #     for j in range(6):
    #         if new_household.iat[i,j] > 1500:
    #             new_household.iat[i,j] = 1500
    #
    # Y=weather.iloc[0:];
    # df2 = pd.DataFrame(Y);
    # df2.to_excel(excel_writer = "final_weather.xlsx", index=False);
    # new_weather=pd.read_excel('final_weather.xlsx');
    # new_weather=new_weather[['DE solar (KW)', 'DE wind (KW)']];
    #
    # rows=len(new_household);
    window=6;
    # store=[0]*6;
    # require=[0]*6;
    # min_threshold=0.2;
    #
    # coal=[0]*6;
    #
    # for i in range(6):
    #     store[i]=min_threshold
    #     if new_weather.iat[0,0] >= new_weather.iat[0,1] and new_weather.iat[0,0] >= min_threshold:
    #         new_weather.iat[0,0]-=min_threshold
    #     elif new_weather.iat[0,1] >= min_threshold :
    #         new_weather.iat[0,1]-=min_threshold
    #     else:
    #         coal[i]+=min_threshold
    #     sum1=0
    #     for j in range(window):
    #         sum1 = sum1 + new_household.iat[j+1,i]
    #     require[i]=sum1
    #
    # content=["Residential No.","Hour No.","Coal","Store","Household Req","Solar","Wind"]
    # for item in content :
    #     worksheet.write(row, column, item)
    #     column += 1
    # row+=1
    # column=0
    #
    # p=-1;
    # for i in range(2530): #time(hours)
    #     prevCoal=[0]*6
    #     solarCons=0
    #     windCons=0
    #     sFlag=[0]*6
    #     wFlag=[0]*6
    #     for j in range(6):
    #         prevCoal[j]=coal[j]
    #     for th in range(6): #house
    #         #store[th]=min_threshold
    #         bal=min_threshold-store[th]
    #         if new_weather.iat[i,0] >= new_weather.iat[i,1] and new_weather.iat[i,0] >= bal:
    #             new_weather.iat[i,0]-=bal
    #             sFlag[th]=bal
    #         elif new_weather.iat[i,1] >= bal :
    #             new_weather.iat[i,1]-=bal
    #             wFlag[th]=bal
    #         else:
    #             coal[th]+=bal
    #     for j in range(6): #houses
    #         if(sFlag[j]>0):
    #             solarCons=sFlag[j]
    #             windCons=0
    #         elif(wFlag[j]>0):
    #             solarCons=0
    #             windCons=wFlag[j]
    #         else:
    #             solarCons=0
    #             windCons-0
    #
    #         if store[j]>= new_household.iat[i,j]:
    #             store[j]-=new_household.iat[i,j]
    #         else:
    #             #do_something
    #             minn=sys.maxsize
    #             for k in range(6): #houses
    #                 if k==j:
    #                     continue
    #                 elif require[k]<minn:
    #                     minn=require[k]
    #                     p=k
    #             #print(minn)
    #             # print(p)
    #             if store[p]>=new_household.iat[i,j]:
    #                 store[p]-=new_household.iat[i,j]
    #                 store[p]+=store[j]
    #                 store[j]=0
    #             else:
    #                 if new_weather.iat[i,0]< new_household.iat[i,j] and new_weather.iat[i,1]<new_household.iat[i,j]:
    #                     coal[j]+=new_household.iat[i,j]
    #                     coal[j]-=store[j]
    #                     store[j]=0
    #                 else:
    #                     if new_weather.iat[i,0] > new_weather.iat[i,1]:
    #                         if new_weather.iat[i,0] > new_household.iat[i,j]:
    #                             new_weather.iat[i,0]-=new_household.iat[i,j]
    #                             solarCons+=new_household.iat[i,j]
    #                             new_weather.iat[i,0]+=store[j]
    #                             solarCons-=store[j]
    #                             store[j]=0
    #                         else:
    #                             coal[j]+=(new_household.iat[i,j]-store[j])
    #                             store[j]=0
    #                     else:
    #                         if new_weather.iat[i,1] > new_household.iat[i,j]:
    #                             new_weather.iat[i,1]-=new_household.iat[i,j]
    #                             windCons+=new_household.iat[i,j]
    #                             new_weather.iat[i,1]+=store[j]
    #                             windCons-=store[j]
    #                             store[j]=0;
    #                         else:
    #                             coal[j]+=(new_household.iat[i,j]-store[j])
    #                             store[j]=0
    #
    #         content=[j+1,i+1,coal[j]-prevCoal[j],store[j],new_household.iat[i,j],solarCons,windCons]
    #         for item in content :
    #             worksheet.write(row, column, item)
    #             column += 1
    #         row+=1
    #         column=0
    #         coalFrame.iat[i,j]=coal[j]
    #             #print(store[j])
    #             #print(new_household.iat[i,j])
    #             #print(new_weather.iat[i,0])
    #             #print(new_weather.iat[i,1])
    #             #print("\n")
    #     row+=1
    #     for abc in range(6):
    #       require[abc]-=new_household.iat[i+1,abc]
    #       require[abc]+=new_household.iat[i+1+window,abc]
    #
    #
    # coalFrame.to_excel(excel_writer = "coalFrame.xlsx", index=False);
    # workbook.close()
    data=Prediction(window=window);
    data.save()
    return render(request,'major/uploadsuccess.html',{'data' : data})

def result(request):
    ddate=request.POST.get('ddate')
    resi=request.POST.get('resi')
    ddate=datetime.strptime(ddate, '%Y-%m-%d')
    d =ddate.strftime("%d")
    m=ddate.strftime("%m")
    y=ddate.strftime("%Y")

    print(d + " " + m + " " + y);
    odate=datetime.strptime('27-10-2016','%d-%m-%Y')
    diff = ddate - odate
    days = diff.days
    days_to_hours = days * 24
    overall_hours = days_to_hours
    print (str(overall_hours) + ' hours');

    row=((overall_hours)*7)+ (91) + int(resi) - (1);
    print(row);

    #for i in range(row , row+(24*7)+1):

    context={}
    data=Prediction(ddate=ddate,resi=resi)
    data.save()
    context['row']=int(row)
    context['resi']=int(resi)
    # rs='major/result/' + str(row) + "/"
    # return HttpResponseRedirect(rs)
    result_file=pd.read_excel('C:/Users/aryak/Desktop/website/result.xlsx');

    m=int(row)
    n=0;
    values=[]
    for ii in range(0,24):
        values.append([])
        for jj in range(0,5):
            n=jj+2
            values[ii].append(round(result_file.iat[m,n],2))
            #values[ii][jj]=(result_file.iat[m,n])
        m+=7;

    coal_consp=0
    solar_consp=0
    wind_consp=0
    for ii in range(0,24):
        coal_consp+=values[ii][0]
        solar_consp+=values[ii][3]
        wind_consp+=values[ii][4]


    context['coal_consp']=round(coal_consp,2)
    context['solar_consp']=round(solar_consp,2)
    context['wind_consp']=round(wind_consp,2)
    context['values']=values
    #print(values)
    return render(request,'major/result.html',context)


def monitor(request):
    pred= Prediction.objects.all()
    context={'pred' : pred}
    return render(request,'major/monitor.html',context)


def moutput1(request):
    finalHousehold=pd.read_excel('final_household.xlsx');
    finalHousehold = finalHousehold.replace(np.nan, 0)
    for i in range(1,8545):
        for j in range(53):
            if(finalHousehold.iat[i,j]==0):
                    finalHousehold.iat[i,j]=finalHousehold.iat[i-1,j];

    subHousehold=finalHousehold.iloc[6001:8531]
    subHousehold.reset_index(inplace = True, drop = True)
    coalFrame=pd.read_excel('coalFrame.xlsx')
    coal_h1 = coalFrame[0]
    initialHour=12
    dishwasher=[0]*24
    freezer=[0]*24
    heatpump=[0]*24
    washing=[0]*24
    for i in range(1,2530):
        if(coal_h1[i]-coal_h1[i-1]>0):
            dishwasher_usage = subHousehold.iat[i,1]-subHousehold.iat[i-1,1]
            freezer_usage = subHousehold.iat[i,2]-subHousehold.iat[i-1,2]
            heatpump_usage = subHousehold.iat[i,4]-subHousehold.iat[i-1,4]
            washing_usage = subHousehold.iat[i,6]-subHousehold.iat[i-1,6]
            maxVal = max(dishwasher_usage,freezer_usage,heatpump_usage,washing_usage)
            if(maxVal==dishwasher_usage):
                #print("Dishwasher")
                hourVal = (initialHour+i-1)%24;
                dishwasher[hourVal]=dishwasher[hourVal]+1;
            elif(maxVal==freezer_usage):
                #print("Dishwasher")
                hourVal = (initialHour+i-1)%24;
                freezer[hourVal]=freezer[hourVal]+1;
            elif(maxVal==heatpump_usage):
                #print("Heat Pump")
                hourVal = (initialHour+i-1)%24;
                heatpump[hourVal]=heatpump[hourVal]+1;
            else:
                #print("Washing Machine")
                hourVal = (initialHour+i-1)%24;
                washing[hourVal]=washing[hourVal]+1;

    #file1 = open("monitorHouse1.txt", "w")
    s=""
    for i in range(24):
        if(dishwasher[i]>0):
            s+="Limit usage of Dishwasher at "
            s+=str(i)
            s+= ":00 hours \n"
        if(freezer[i]>0):
            s+="Limit usage of Freezer at "
            s+=str(i)
            s+=":00 hours \n"
        if(heatpump[i]>0):
            s+="Limit usage of Heat Pump at "
            s+=str(i)
            s+=":00 hours \n"
        if(washing[i]>0):
            s+="Limit usage of Washing Machine at "
            s+=str(i)
            s+=":00 hours \n"

    #file1.close()

    context={}
    context['s']=s
    return render(request,'major/moutput.html',context)


def moutput2(request):
    finalHousehold=pd.read_excel('final_household.xlsx');
    finalHousehold = finalHousehold.replace(np.nan, 0)
    for i in range(1,8545):
        for j in range(53):
            if(finalHousehold.iat[i,j]==0):
                    finalHousehold.iat[i,j]=finalHousehold.iat[i-1,j];

    subHousehold=finalHousehold.iloc[6001:8531]
    subHousehold.reset_index(inplace = True, drop = True)
    coalFrame=pd.read_excel('coalFrame.xlsx')
    coal_h2 = coalFrame[1]
    initialHour=12
    ciculation=[0]*24
    dishwasher=[0]*24
    freezer=[0]*24
    heatpump=[0]*24
    washing=[0]*24
    for i in range(1,2530):
        if(coal_h2[i]-coal_h2[i-1]>0):
            circulation_usage = subHousehold.iat[i,9]-subHousehold.iat[i-1,9]
            dishwasher_usage = subHousehold.iat[i,10]-subHousehold.iat[i-1,10]
            freezer_usage = subHousehold.iat[i,11]-subHousehold.iat[i-1,11]
            heatpump_usage = subHousehold.iat[i,13]-subHousehold.iat[i-1,13]
            washing_usage = subHousehold.iat[i,14]-subHousehold.iat[i-1,14]
            maxVal = max(circulation_usage,dishwasher_usage,freezer_usage,heatpump_usage,washing_usage)
            if(maxVal==circulation_usage):
                #print("Circulation Pump")
                hourVal = (initialHour+i-1)%24;
                ciculation[hourVal]=ciculation[hourVal]+1;
            elif(maxVal==dishwasher_usage):
                #print("Dishwasher")
                hourVal = (initialHour+i-1)%24;
                dishwasher[hourVal]=dishwasher[hourVal]+1;
            elif(maxVal==freezer_usage):
                #print("Dishwasher")
                hourVal = (initialHour+i-1)%24;
                freezer[hourVal]=freezer[hourVal]+1;
            elif(maxVal==heatpump_usage):
                #print("Heat Pump")
                hourVal = (initialHour+i-1)%24;
                heatpump[hourVal]=heatpump[hourVal]+1;
            else:
                #print("Washing Machine")
                hourVal = (initialHour+i-1)%24;
                washing[hourVal]=washing[hourVal]+1;

    #file1 = open("monitorHouse1.txt", "w")
    s=""
    for i in range(24):
        if(ciculation[i]>0):
            s+="Limit usage of Circulation Pump at "
            s+=str(i)
            s+= ":00 hours \n"
        if(dishwasher[i]>0):
            s+="Limit usage of Dishwasher at "
            s+=str(i)
            s+= ":00 hours \n"
        if(freezer[i]>0):
            s+="Limit usage of Freezer at "
            s+=str(i)
            s+=":00 hours \n"
        if(heatpump[i]>0):
            s+="Limit usage of Heat Pump at "
            s+=str(i)
            s+=":00 hours \n"
        if(washing[i]>0):
            s+="Limit usage of Washing Machine at "
            s+=str(i)
            s+=":00 hours \n"

    #file1.close()

    context={}
    context['s']=s
    return render(request,'major/moutput.html',context)

def moutput3(request):
    finalHousehold=pd.read_excel('final_household.xlsx');
    finalHousehold = finalHousehold.replace(np.nan, 0)
    for i in range(1,8545):
        for j in range(53):
            if(finalHousehold.iat[i,j]==0):
                    finalHousehold.iat[i,j]=finalHousehold.iat[i-1,j];

    subHousehold=finalHousehold.iloc[6001:8531]
    subHousehold.reset_index(inplace = True, drop = True)
    coalFrame=pd.read_excel('coalFrame.xlsx')
    coal_h3 = coalFrame[2]
    initialHour=12
    ciculation=[0]*24
    dishwasher=[0]*24
    freezer=[0]*24
    refrigerator=[0]*24
    washing=[0]*24
    for i in range(1,2530):
        if(coal_h3[i]-coal_h3[i-1]>0):
            circulation_usage = subHousehold.iat[i,17]-subHousehold.iat[i-1,17]
            dishwasher_usage = subHousehold.iat[i,18]-subHousehold.iat[i-1,18]
            freezer_usage = subHousehold.iat[i,19]-subHousehold.iat[i-1,19]
            refrigerator_usage = subHousehold.iat[i,23]-subHousehold.iat[i-1,23]
            washing_usage = subHousehold.iat[i,24]-subHousehold.iat[i-1,24]
            maxVal = max(circulation_usage,dishwasher_usage,freezer_usage,refrigerator_usage,washing_usage)
            if(maxVal==circulation_usage):
                #print("Circulation Pump")
                hourVal = (initialHour+i-1)%24;
                ciculation[hourVal]=ciculation[hourVal]+1;
            elif(maxVal==dishwasher_usage):
                #print("Dishwasher")
                hourVal = (initialHour+i-1)%24;
                dishwasher[hourVal]=dishwasher[hourVal]+1;
            elif(maxVal==freezer_usage):
                #print("Dishwasher")
                hourVal = (initialHour+i-1)%24;
                freezer[hourVal]=freezer[hourVal]+1;
            elif(maxVal==refrigerator_usage):
                #print("Heat Pump")
                hourVal = (initialHour+i-1)%24;
                refrigerator[hourVal]=refrigerator[hourVal]+1;
            else:
                #print("Washing Machine")
                hourVal = (initialHour+i-1)%24;
                washing[hourVal]=washing[hourVal]+1;
    s=""
    for i in range(24):
        if(ciculation[i]>0):
            s+="Limit usage of Circulation Pump at "
            s+=str(i)
            s+= ":00 hours \n"
        if(dishwasher[i]>0):
            s+="Limit usage of Dishwasher at "
            s+=str(i)
            s+= ":00 hours \n"
        if(freezer[i]>0):
            s+="Limit usage of Freezer at "
            s+=str(i)
            s+=":00 hours \n"
        if(refrigerator[i]>0):
            s+="Limit usage of Refrigerator at "
            s+=str(i)
            s+=":00 hours \n"
        if(washing[i]>0):
            s+="Limit usage of Washing Machine at "
            s+=str(i)
            s+=":00 hours \n"


    context={}
    context['s']=s
    return render(request,'major/moutput.html',context)


def moutput4(request):
    finalHousehold=pd.read_excel('final_household.xlsx');
    finalHousehold = finalHousehold.replace(np.nan, 0)
    for i in range(1,8545):
        for j in range(53):
            if(finalHousehold.iat[i,j]==0):
                    finalHousehold.iat[i,j]=finalHousehold.iat[i-1,j];

    subHousehold=finalHousehold.iloc[6001:8531]
    subHousehold.reset_index(inplace = True, drop = True)
    coalFrame=pd.read_excel('coalFrame.xlsx')
    coal_h4 = coalFrame[3]
    initialHour=12
    dishwasher=[0]*24
    ev=[0]*24
    freezer=[0]*24
    heatpump=[0]*24
    refrigerator=[0]*24
    washing=[0]*24
    for i in range(1,2530):
        if(coal_h4[i]-coal_h4[i-1]>0):
            dishwasher_usage = subHousehold.iat[i,27]-subHousehold.iat[i-1,27]
            ev_usage = subHousehold.iat[i,28]-subHousehold.iat[i-1,28]
            freezer_usage = subHousehold.iat[i,29]-subHousehold.iat[i-1,29]
            heatpump_usage = subHousehold.iat[i,32]-subHousehold.iat[i-1,32]
            refrigerator_usage = subHousehold.iat[i,34]-subHousehold.iat[i-1,34]
            washing_usage = subHousehold.iat[i,35]-subHousehold.iat[i-1,35]
            maxVal = max(dishwasher_usage,ev_usage,freezer_usage,heatpump_usage,refrigerator_usage,washing_usage)
            if(maxVal==dishwasher_usage):
                #print("Dishwasher")
                hourVal = (initialHour+i-1)%24;
                dishwasher[hourVal]=dishwasher[hourVal]+1;
            elif(maxVal==ev_usage):
                #print("Dishwasher")
                hourVal = (initialHour+i-1)%24;
                ev[hourVal]=ev[hourVal]+1;
            elif(maxVal==freezer_usage):
                #print("Dishwasher")
                hourVal = (initialHour+i-1)%24;
                freezer[hourVal]=freezer[hourVal]+1;
            elif(maxVal==heatpump_usage):
                #print("Dishwasher")
                hourVal = (initialHour+i-1)%24;
                heatpump[hourVal]=heatpump[hourVal]+1;
            elif(maxVal==refrigerator_usage):
                #print("Heat Pump")
                hourVal = (initialHour+i-1)%24;
                refrigerator[hourVal]=refrigerator[hourVal]+1;
            else:
                #print("Washing Machine")
                hourVal = (initialHour+i-1)%24;
                washing[hourVal]=washing[hourVal]+1;
    s=""
    for i in range(24):
        if(dishwasher[i]>0):
            s+="Limit usage of Dishwasher at "
            s+=str(i)
            s+= ":00 hours \n"
        if(ev[i]>0):
            s+="Limit usage of Circulation Pump at "
            s+=str(i)
            s+= ":00 hours \n"
        if(freezer[i]>0):
            s+="Limit usage of Freezer at "
            s+=str(i)
            s+=":00 hours \n"
        if(heatpump[i]>0):
            s+="Limit usage of Heat Pump at "
            s+=str(i)
            s+=":00 hours \n"
        if(refrigerator[i]>0):
            s+="Limit usage of Refrigerator at "
            s+=str(i)
            s+=":00 hours \n"
        if(washing[i]>0):
            s+="Limit usage of Washing Machine at "
            s+=str(i)
            s+=":00 hours \n"


    context={}
    context['s']=s
    return render(request,'major/moutput.html',context)

def moutput5(request):
    finalHousehold=pd.read_excel('final_household.xlsx');
    finalHousehold = finalHousehold.replace(np.nan, 0)
    for i in range(1,8545):
        for j in range(53):
            if(finalHousehold.iat[i,j]==0):
                    finalHousehold.iat[i,j]=finalHousehold.iat[i-1,j];

    subHousehold=finalHousehold.iloc[6001:8531]
    subHousehold.reset_index(inplace = True, drop = True)
    coalFrame=pd.read_excel('coalFrame.xlsx')
    coal_h5 = coalFrame[4]
    initialHour=12
    dishwasher=[0]*24
    refrigerator=[0]*24
    washing=[0]*24
    for i in range(1,2530):
        if(coal_h5[i]-coal_h5[i-1]>0):
            dishwasher_usage = subHousehold.iat[i,38]-subHousehold.iat[i-1,38]
            refrigerator_usage = subHousehold.iat[i,40]-subHousehold.iat[i-1,40]
            washing_usage = subHousehold.iat[i,41]-subHousehold.iat[i-1,41]
            maxVal = max(dishwasher_usage,refrigerator_usage,washing_usage)
            if(maxVal==dishwasher_usage):
                #print("Dishwasher")
                hourVal = (initialHour+i-1)%24;
                dishwasher[hourVal]=dishwasher[hourVal]+1;
            elif(maxVal==refrigerator_usage):
                #print("Heat Pump")
                hourVal = (initialHour+i-1)%24;
                refrigerator[hourVal]=refrigerator[hourVal]+1;
            else:
                #print("Washing Machine")
                hourVal = (initialHour+i-1)%24;
                washing[hourVal]=washing[hourVal]+1;
    s=""
    for i in range(24):
        if(dishwasher[i]>0):
            s+="Limit usage of Dishwasher at "
            s+=str(i)
            s+= ":00 hours \n"
        if(refrigerator[i]>0):
            s+="Limit usage of Refrigerator at "
            s+=str(i)
            s+=":00 hours \n"
        if(washing[i]>0):
            s+="Limit usage of Washing Machine at "
            s+=str(i)
            s+=":00 hours \n"


    context={}
    context['s']=s
    return render(request,'major/moutput.html',context)


def moutput6(request):
    finalHousehold=pd.read_excel('final_household.xlsx');
    finalHousehold = finalHousehold.replace(np.nan, 0)
    for i in range(1,8545):
        for j in range(53):
            if(finalHousehold.iat[i,j]==0):
                    finalHousehold.iat[i,j]=finalHousehold.iat[i-1,j];

    subHousehold=finalHousehold.iloc[6001:8531]
    subHousehold.reset_index(inplace = True, drop = True)
    coalFrame=pd.read_excel('coalFrame.xlsx')
    coal_h6 = coalFrame[5]
    initialHour=12
    ciculation=[0]*24
    dishwasher=[0]*24
    freezer=[0]*24
    washing=[0]*24
    for i in range(1,2530):
        if(coal_h6[i]-coal_h6[i-1]>0):
            circulation_usage = subHousehold.iat[i,44]-subHousehold.iat[i-1,44]
            dishwasher_usage = subHousehold.iat[i,45]-subHousehold.iat[i-1,45]
            freezer_usage = subHousehold.iat[i,46]-subHousehold.iat[i-1,46]
            washing_usage = subHousehold.iat[i,50]-subHousehold.iat[i-1,50]
            maxVal = max(circulation_usage,dishwasher_usage,freezer_usage,washing_usage)
            if(maxVal==circulation_usage):
                #print("Dishwasher")
                hourVal = (initialHour+i-1)%24;
                ciculation[hourVal]=ciculation[hourVal]+1;
            elif(maxVal==dishwasher_usage):
                #print("Dishwasher")
                hourVal = (initialHour+i-1)%24;
                dishwasher[hourVal]=dishwasher[hourVal]+1;
            elif(maxVal==freezer_usage):
                #print("Heat Pump")
                hourVal = (initialHour+i-1)%24;
                freezer[hourVal]=freezer[hourVal]+1;
            else:
                #print("Washing Machine")
                hourVal = (initialHour+i-1)%24;
                washing[hourVal]=washing[hourVal]+1;
    s=""
    for i in range(24):
        if(ciculation[i]>0):
            s+="Limit usage of Circulation Pump at "
            s+=str(i)
            s+= ":00 hours \n"
        if(dishwasher[i]>0):
            s+="Limit usage of Dishwasher at "
            s+=str(i)
            s+= ":00 hours \n"
        if(freezer[i]>0):
            s+="Limit usage of Refrigerator at "
            s+=str(i)
            s+=":00 hours \n"
        if(washing[i]>0):
            s+="Limit usage of Washing Machine at "
            s+=str(i)
            s+=":00 hours \n"


    context={}
    context['s']=s
    return render(request,'major/moutput.html',context)
