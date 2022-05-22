import pandas
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


def calculate_loss(name, col):
    data = pandas.read_csv(name)
    train = data[12:42]
    test = data[0:12]
    poly_loss(name, col, train, test)
    svr_loss(name, col, train, test)


def poly_loss(name, col, train, test):
    x = numpy.array(range(30)).reshape(-1, 1)
    test_x = numpy.array(range(30, 42)).reshape(-1, 1)
    poly_reg = PolynomialFeatures(degree=2)
    x_poly = poly_reg.fit_transform(x)
    test_x_poly = poly_reg.fit_transform(test_x)
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(x_poly, train[col])
    predict = poly_reg_model.predict(test_x_poly)

    loss = 0
    for i in range(12):
        loss += abs(predict[i] - test.loc[11 - i, col])
    print(str(name) + '\t' + str(col) + '\t多项式回归:loss = ' + str(loss))


def svr_loss(name, col, train, test):
    x = numpy.array(range(30)).reshape(-1, 1)
    test_x = numpy.array(range(30, 42)).reshape(-1, 1)
    scaled_x = StandardScaler()
    scaled_y = StandardScaler()
    scaled_x_test = StandardScaler()
    scaled_x = scaled_x.fit_transform(x)
    scaled_y = scaled_y.fit_transform(numpy.array(train[col]).reshape(-1, 1))
    scaled_x_test = scaled_x_test.fit_transform(test_x)
    svr_model = SVR(kernel='rbf', gamma='auto')
    svr_model.fit(scaled_x, scaled_y.ravel())
    predict = svr_model.predict(scaled_x_test)

    loss = 0
    for i in range(12):
        loss += abs(predict[i] - test.loc[11 - i, col])
    print(str(name) + '\t' + str(col) + '\t支持向量机回归:loss = ' + str(loss))


if __name__ == '__main__':
    calculate_loss('beijing_data.csv', 'confirmedNum')
    calculate_loss('beijing_data.csv', 'curesNum')
    calculate_loss('changchun_data.csv', 'confirmedNum')
    calculate_loss('changchun_data.csv', 'curesNum')
    calculate_loss('shanghai_data.csv', 'confirmedNum')
    calculate_loss('shanghai_data.csv', 'curesNum')
    calculate_loss('shenzhen_data.csv', 'confirmedNum')
    calculate_loss('shenzhen_data.csv', 'curesNum')
