import pandas as pd;
import numpy as np;
finalHousehold=pd.read_excel('final_household.xlsx');

finalHousehold = finalHousehold.replace(np.nan, 0)
for i in range(1,8545):
    for j in range(53):
        if(finalHousehold.iat[i,j]==0):
                finalHousehold.iat[i,j]=finalHousehold.iat[i-1,j];

subHousehold=finalHousehold.iloc[6001:8531]
subHousehold.reset_index(inplace = True, drop = True)
coalFrame=pd.read_excel('coalFrame.xlsx')

# Household 1

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

file1 = open("monitorHouse1.txt", "w")

for i in range(24):
    if(dishwasher[i]>0):
        file1.write("Limit usage of Dishwasher at ")
        file1.write(str(i))
        file1.write(":00 hours \n")
    if(freezer[i]>0):
        file1.write("Limit usage of Freezer at ")
        file1.write(str(i))
        file1.write(":00 hours \n")
    if(heatpump[i]>0):
        file1.write("Limit usage of Heat Pump at ")
        file1.write(str(i))
        file1.write(":00 hours \n")
    if(washing[i]>0):
        file1.write("Limit usage of Washing Machine at ")
        file1.write(str(i))
        file1.write(":00 hours \n")

file1.close()

# Household 2

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

file2 = open("monitorHouse2.txt", "w")

for i in range(24):
    if(ciculation[i]>0):
        file2.write("Limit usage of Circulation Pump at ")
        file2.write(str(i))
        file2.write(":00 hours \n")
    if(dishwasher[i]>0):
        file2.write("Limit usage of Dishwasher at ")
        file2.write(str(i))
        file2.write(":00 hours \n")
    if(freezer[i]>0):
        file2.write("Limit usage of Freezer at ")
        file2.write(str(i))
        file2.write(":00 hours \n")
    if(heatpump[i]>0):
        file2.write("Limit usage of Heat Pump at ")
        file2.write(str(i))
        file2.write(":00 hours \n")
    if(washing[i]>0):
        file2.write("Limit usage of Washing Machine at ")
        file2.write(str(i))
        file2.write(":00 hours \n")

file2.close()

# Household 3

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

file3 = open("monitorHouse3.txt", "w")

for i in range(24):
    if(ciculation[i]>0):
        file3.write("Limit usage of Circulation Pump at ")
        file3.write(str(i))
        file3.write(":00 hours \n")
    if(dishwasher[i]>0):
        file3.write("Limit usage of Dishwasher at ")
        file3.write(str(i))
        file3.write(":00 hours \n")
    if(freezer[i]>0):
        file3.write("Limit usage of Freezer at ")
        file3.write(str(i))
        file3.write(":00 hours \n")
    if(refrigerator[i]>0):
        file3.write("Limit usage of Refrigerator at ")
        file3.write(str(i))
        file3.write(":00 hours \n")
    if(washing[i]>0):
        file3.write("Limit usage of Washing Machine at ")
        file3.write(str(i))
        file3.write(":00 hours \n")

file3.close()

# Household 4

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

file4 = open("monitorHouse4.txt", "w")

for i in range(24):
    if(dishwasher[i]>0):
        file4.write("Limit usage of Dishwasher at ")
        file4.write(str(i))
        file4.write(":00 hours \n")
    if(ev[i]>0):
        file4.write("Limit usage of Electric Vehicle (charging) at ")
        file4.write(str(i))
        file4.write(":00 hours \n")
    if(freezer[i]>0):
        file4.write("Limit usage of Freezer at ")
        file4.write(str(i))
        file4.write(":00 hours \n")
    if(heatpump[i]>0):
        file4.write("Limit usage of Heat Pump at ")
        file4.write(str(i))
        file4.write(":00 hours \n")
    if(refrigerator[i]>0):
        file4.write("Limit usage of Refrigerator at ")
        file4.write(str(i))
        file4.write(":00 hours \n")
    if(washing[i]>0):
        file4.write("Limit usage of Washing Machine at ")
        file4.write(str(i))
        file4.write(":00 hours \n")

file4.close()

# Household 5

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

file5 = open("monitorHouse5.txt", "w")

for i in range(24):
    if(dishwasher[i]>0):
        file5.write("Limit usage of Dishwasher at ")
        file5.write(str(i))
        file5.write(":00 hours \n")
    if(refrigerator[i]>0):
        file5.write("Limit usage of Refrigerator at ")
        file5.write(str(i))
        file5.write(":00 hours \n")
    if(washing[i]>0):
        file5.write("Limit usage of Washing Machine at ")
        file5.write(str(i))
        file5.write(":00 hours \n")

file5.close()

# Household 6

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

file6 = open("monitorHouse6.txt", "w")

for i in range(24):
    if(ciculation[i]>0):
        file6.write("Limit usage of Circulation Pump at ")
        file6.write(str(i))
        file6.write(":00 hours \n")
    if(dishwasher[i]>0):
        file6.write("Limit usage of Dishwasher at ")
        file6.write(str(i))
        file6.write(":00 hours \n")
    if(freezer[i]>0):
        file6.write("Limit usage of Freezer at ")
        file6.write(str(i))
        file6.write(":00 hours \n")
    if(washing[i]>0):
        file6.write("Limit usage of Washing Machine at ")
        file6.write(str(i))
        file6.write(":00 hours \n")

file6.close()
