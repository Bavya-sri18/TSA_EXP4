# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 11.05.2026

### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:

```PYTHON
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv("/content/VAPO.csv")

df['Date'] = pd.to_datetime(df['Date'])

df.set_index('Date', inplace=True)

X = df['Close']

N = 1000

plt.rcParams['figure.figsize'] = [12, 6]

plt.plot(X)
plt.title('Original VAPO Data')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

plt.subplot(2, 1, 1)
plot_acf(X, lags=30, ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=30, ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_1)
plt.title('ACF of ARMA(1,1)')
plt.show()

plot_pacf(ARMA_1)
plt.title('PACF of ARMA(1,1)')
plt.show()


arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']

theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N * 10)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_2)
plt.title('ACF of ARMA(2,2)')
plt.show()

plot_pacf(ARMA_2)
plt.title('PACF of ARMA(2,2)')
plt.show()
```

### OUTPUT:

<img width="1142" height="622" alt="image" src="https://github.com/user-attachments/assets/5801e7a4-39f5-495f-836c-7cb2252508d9" />


<img width="1367" height="347" alt="image" src="https://github.com/user-attachments/assets/9dd464e1-d741-4731-b3b9-527dc637878a" />


<img width="1387" height="336" alt="image" src="https://github.com/user-attachments/assets/7e057e78-b455-4761-b2cd-5cdd63a7aea3" />



#### SIMULATED ARMA(1,1) PROCESS:

<img width="1155" height="592" alt="image" src="https://github.com/user-attachments/assets/0cd1d50a-a625-475a-ba67-4a701ebe1067" />

#### Autocorrelation

<img width="1172" height="606" alt="image" src="https://github.com/user-attachments/assets/bfcd461c-5370-4f40-8483-c4c824ff942e" />

#### Partial Autocorrelation

<img width="1146" height="592" alt="image" src="https://github.com/user-attachments/assets/12e82bc5-f58c-40ff-8e6d-3b88758ee81e" />


#### SIMULATED ARMA(2,2) PROCESS:

<img width="1152" height="592" alt="image" src="https://github.com/user-attachments/assets/fc5d6a08-464a-4cdd-9cf4-006b0b3bebaf" />

#### Autocorrelation

<img width="1187" height="612" alt="image" src="https://github.com/user-attachments/assets/b84ed05a-cdf4-45f8-b82d-4c11efff39ac" />

#### Partial Autocorrelation

<img width="1157" height="597" alt="image" src="https://github.com/user-attachments/assets/20e60cff-aff5-46b9-9118-4dbc3cd4b3e9" />


### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
