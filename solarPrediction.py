import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import LSTM
df=pd.read_excel('dataWeather.xlsx');

dateAr = df['Date']
dateAr = dateAr.loc[6001:]
dateAr2 = pd.DataFrame(dateAr);
#dateAr2.to_excel(excel_writer = "intDate.xlsx", index=False);

print(df.head())
df.drop(['DE wind (KW)'], axis=1, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
ind_df = df.set_index(['Date'], drop=True)
ind_df.head()
ind_df = ind_df.sort_index()
plt.figure(figsize=(10, 6))
# ind_df['DE solar (MW)'].plot();
# split_date = pd.Timestamp('20-11-2016  22:00:00')
# df =  df['DE solar (MW)']
train = df.loc[:6000]
test = df.loc[6001:]
inttr = train.set_index(['Date'], drop=True)
intte = test.set_index(['Date'], drop=True)
plt.figure(figsize=(10, 6))
ax = inttr.plot()
intte.plot(ax=ax)
plt.legend(['train', 'test']);

scaler = MinMaxScaler(feature_range=(-1, 1))
train_sc = scaler.fit_transform(inttr)
test_sc = scaler.transform(intte)
# np.random.shuffle(train_sc)
X_train = train_sc[:-1]
y_train = train_sc[1:]
X_test = test_sc[:-1]
y_test = test_sc[1:]
X_train = np.reshape(X_train, (6000, 1, 1))
X_test = np.reshape(X_test, (2544, 1, 1))
lstm_model = Sequential()
lstm_model.add(LSTM(7, input_shape=(1, X_train.shape[1]), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history_lstm_model = lstm_model.fit(X_train, y_train, epochs=5, batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop])
y_pred_test_lstm = lstm_model.predict(X_test)
y_train_pred_lstm = lstm_model.predict(X_train)
ypf_train = scaler.inverse_transform(y_train_pred_lstm)
ypf_test = scaler.inverse_transform(y_pred_test_lstm)
ypf2 = pd.DataFrame(ypf_test);
ypf2.columns = ['DE solar (KW)']
dateAr2.drop(dateAr2.tail(1).index,inplace=True)
dateAr2['DE solar (KW)']=ypf2.values


dateAr2.to_excel(excel_writer = "dateSolar.xlsx", index=False);



print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_lstm)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_lstm)))
lstm_test_mse = lstm_model.evaluate(X_test, y_test, batch_size=1)
print('LSTM: %f'%lstm_test_mse)
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True')
plt.plot(y_pred_test_lstm, label='LSTM')
plt.title("LSTM's Prediction")
plt.xlabel('Observation')
plt.ylabel('DE solar (KW)')
plt.legend()
plt.show();
