from sklearn import svm,datasets


class Dataset:

    def __init__(self,name):
        self.name = name


    def download_data(self):
        if self.name == 'iris':
            self.downloaded_data = datasets.load_iris()
        elif self.name == 'digits':
            self.downloaded_data = datasets.load_digits()
        else:
            print('Dataset error:No named datasets')

    def generate_xy(self):
        self.download_data()
        x = self.downloaded_data.data
        y = self.downloaded_data.target
        print('\n Original data looks like this:\n',x)
        print('\n Labels looks like this:\n',y)
        return x,y

    def get_train_test_set(self,ratio):
        x,y = self.generate_xy()
        n_samples = len(x)
        n_train = n_samples * ratio
        print('n_train:',n_train)
        n_train = int(n_train)
        print('n_train_2:',n_train)

        x_train = x[:n_train]
        y_train = y[:n_train]

        x_test = x[n_train:]
        y_test = y[n_train:]

        return x_train,y_train,x_test,y_test

if __name__ == '__main__':
    data = Dataset('digits')
    x_train,y_train,x_test,y_test=data.get_train_test_set(0.7)
    print('\n x_trian',x_train)
    print('\n y_train',y_train)

    clf = svm.SVC()
    clf.fit(x_train,y_train)

    test_point = x_test[12]
    y_true = y_test[12]

    y_p = clf.predict(test_point)

    print('y_true:',y_true)
    print('y_predict:',y_p)
          
    print(y_true == y_p)
    
    
