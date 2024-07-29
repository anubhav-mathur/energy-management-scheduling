import pandas as pd;
import numpy as np;
import xlsxwriter

workbook = xlsxwriter.Workbook('result.xlsx')
worksheet = workbook.add_worksheet()
row = 0
column = 0

import sys;

weather=pd.read_excel('predWeather.xlsx');
household=pd.read_excel('dataHousehold.xlsx');

X=household.iloc[3:];
df = pd.DataFrame(X);
df.to_excel(excel_writer = "final_household.xlsx",header=False,index=False);
new_household=pd.read_excel('final_household.xlsx');
new_household=new_household.loc[6001:]
new_household=new_household[['Total_1','Total_2','Total_3','Total_4','Total_5','Total_6']];

coalFrame = [[0]*6]*2530
coalFrame=pd.DataFrame(coalFrame)
for i in range(2544):
    for j in range(6):
        if new_household.iat[i,j] < 0 :
            new_household.iat[i,j] = 0

for i in range(2544):
    for j in range(6):
        if new_household.iat[i,j] > 1500:
            new_household.iat[i,j] = 1500

Y=weather.iloc[0:];
df2 = pd.DataFrame(Y);
df2.to_excel(excel_writer = "final_weather.xlsx", index=False);
new_weather=pd.read_excel('final_weather.xlsx');
new_weather=new_weather[['DE solar (KW)', 'DE wind (KW)']];

rows=len(new_household);
window=6;
store=[0]*6;
require=[0]*6;
min_threshold=0.2;

coal=[0]*6;

for i in range(6):
    store[i]=min_threshold
    if new_weather.iat[0,0] >= new_weather.iat[0,1] and new_weather.iat[0,0] >= min_threshold:
        new_weather.iat[0,0]-=min_threshold
    elif new_weather.iat[0,1] >= min_threshold :
        new_weather.iat[0,1]-=min_threshold
    else:
        coal[i]+=min_threshold
    sum1=0
    for j in range(window):
        sum1 = sum1 + new_household.iat[j+1,i]
    require[i]=sum1

content=["Residential No.","Hour No.","Coal","Store","Household Req","Solar","Wind"]
for item in content :
    worksheet.write(row, column, item)
    column += 1
row+=1
column=0

p=-1;
for i in range(2530): #time(hours)
    prevCoal=[0]*6
    solarCons=0
    windCons=0
    sFlag=[0]*6
    wFlag=[0]*6
    for j in range(6):
        prevCoal[j]=coal[j]
    for th in range(6): #house
        #store[th]=min_threshold
        bal=min_threshold-store[th]
        if new_weather.iat[i,0] >= new_weather.iat[i,1] and new_weather.iat[i,0] >= bal:
            new_weather.iat[i,0]-=bal
            sFlag[th]=bal
        elif new_weather.iat[i,1] >= bal :
            new_weather.iat[i,1]-=bal
            wFlag[th]=bal
        else:
            coal[th]+=bal
    for j in range(6): #houses
        if(sFlag[j]>0):
            solarCons=sFlag[j]
            windCons=0
        elif(wFlag[j]>0):
            solarCons=0
            windCons=wFlag[j]
        else:
            solarCons=0
            windCons=0

        if store[j]>= new_household.iat[i,j]:
            store[j]-=new_household.iat[i,j]
        else:
            #do_something
            minn=sys.maxsize
            for k in range(6): #houses
                if k==j:
                    continue
                elif require[k]<minn:
                    minn=require[k]
                    p=k
            #print(minn)
            # print(p)
            if store[p]>=new_household.iat[i,j]:
                store[p]-=new_household.iat[i,j]
                store[p]+=store[j]
                store[j]=0
            else:
                if new_weather.iat[i,0]< new_household.iat[i,j] and new_weather.iat[i,1]<new_household.iat[i,j]:
                    coal[j]+=new_household.iat[i,j]
                    coal[j]-=store[j]
                    store[j]=0
                else:
                    if new_weather.iat[i,0] > new_weather.iat[i,1]:
                        if new_weather.iat[i,0] > new_household.iat[i,j]:
                            new_weather.iat[i,0]-=new_household.iat[i,j]
                            solarCons+=new_household.iat[i,j]
                            new_weather.iat[i,0]+=store[j]
                            solarCons-=store[j]
                            store[j]=0
                        else:
                            coal[j]+=(new_household.iat[i,j]-store[j])
                            store[j]=0
                    else:
                        if new_weather.iat[i,1] > new_household.iat[i,j]:
                            new_weather.iat[i,1]-=new_household.iat[i,j]
                            windCons+=new_household.iat[i,j]
                            new_weather.iat[i,1]+=store[j]
                            windCons-=store[j]
                            store[j]=0;
                        else:
                            coal[j]+=(new_household.iat[i,j]-store[j])
                            store[j]=0

        content=[j+1,i+1,coal[j]-prevCoal[j],store[j],new_household.iat[i,j],solarCons,windCons]
        for item in content :
            worksheet.write(row, column, item)
            column += 1
        row+=1
        column=0
        coalFrame.iat[i,j]=coal[j]
            #print(store[j])
            #print(new_household.iat[i,j])
            #print(new_weather.iat[i,0])
            #print(new_weather.iat[i,1])
            #print("\n")
    row+=1
    for abc in range(6):
      require[abc]-=new_household.iat[i+1,abc]
      require[abc]+=new_household.iat[i+1+window,abc]


coalFrame.to_excel(excel_writer = "coalFrame.xlsx", index=False);
workbook.close()
