import pandas as pd
import numpy as np
from sklearn.svm import SVR
from matplotlib import pyplot as plt


def predict_price(dt, pr, x):

    dt = dt.values.reshape(len(dt), 1)

    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)

    svr_rbf.fit(dt, pr)
    svr_poly.fit(dt, pr)
    #svr_lin.fit(dt, pr)

    plt.axis([0, 300, 0, 250])
    plt.scatter(dt[:, 0], prices, color='black', label='data') # plot dei punti del dataset
    plt.plot(dates, svr_rbf.predict(dt), color='red', label='RBF') # plot delle predizioni della svm
    #plt.plot(dates, svr_lin.predict(dt), color='green', label='Lin')
    plt.plot(dates, svr_poly.predict(dt), color='blue', label='Poly')

    plt.xlabel("date")
    plt.ylabel("Price")
    plt.title("Support Vector Regression")
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_poly.predict(x)[0]


def predict_price_multi(dt, pr, x):

    #dt = dt.values.reshape(len(dt), 1)

    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)

    svr_rbf.fit(dt, pr)
    svr_poly.fit(dt, pr)
    #svr_lin.fit(dt, pr)

    plt.axis([0, 300, 0, 250])
    plt.scatter(dt[:, 0], prices, color='black', label='data') # plot dei punti del dataset
    plt.plot(dates, svr_rbf.predict(dt), color='red', label='RBF') # plot delle predizioni della svm
    #plt.plot(dates, svr_lin.predict(dt), color='green', label='Lin')
    plt.plot(dates, svr_poly.predict(dt), color='blue', label='Poly')

    plt.xlabel("date")
    plt.ylabel("Price")
    plt.title("Support Vector Regression")
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_poly.predict(x)[0]


apple_data = pd.read_csv("AAPL.csv")
apple_data['date'] = apple_data['Date'].map(lambda x: str(x).replace("-", ""))
print(apple_data.head())
apple_data.drop('Date', inplace=True, axis=1)
print(apple_data.head())
print(apple_data.shape)

#cols = ['date','High','Low','Close', 'Close']
#cols = ['date']
#dates = apple_data[cols]
dates = apple_data['date']
prices = apple_data['Open']
print(prices.head())

pred1, pred2 = predict_price(dates, prices, 20180714)

#pred1, pred2 = predict_price_multi(dates, prices, 20180714)

print(pred1)
print(pred2)
