import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, VarianceThreshold, chi2
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV


columns = ['lettr', 'x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar', 'y-bar', 'x2bar',
           'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx']
df = pd.read_csv('./letter-recognition.data.txt', encoding='utf-8', header=None, names=columns)


# Binary Classification
def preprocess(data, trainSize=0.9):
    ix = int(data.shape[0] * trainSize)
    train = data[:ix]
    test = data[ix:]
    train_X, train_y = train.to_numpy()[:, 1:], train.to_numpy()[:, 0]
    test_X, test_y = test.to_numpy()[:, 1:], test.to_numpy()[:, 0]
    return train_X, train_y, test_X, test_y


def createPairs(df):
    hk = df.loc[(df.lettr == 'H') | (df.lettr == 'K')]
    my = df.loc[(df.lettr == 'M') | (df.lettr == 'Y')]
    rv = df.loc[(df.lettr == 'R') | (df.lettr == 'V')]
    return {
        'hk': preprocess(hk),
        'my': preprocess(my),
        'rv': preprocess(rv),
    }


def dimensionReduction(train_X, test_X, train_y, dt=None):
    if dt == 'hk':
        th = 4.9
    elif dt == 'my':
        th = 7.5
    elif dt == 'rv':
        th = 4.7
    else:  # None is for the multiple classification
        th = 4
    var = VarianceThreshold(threshold=th)  # Simple Quality Filtering
    chi = SelectKBest(chi2, k=4)  # Filter Methods
    rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=4)  # Wrapper Feature Selection
    svc = SelectFromModel(estimator=LinearSVC(), max_features=4)  # Embedded Methods
    pca = PCA(n_components=4)  # Feature Extraction

    return {
        'None': [
            train_X,
            test_X
        ],
        'simpleQuality': [
            var.fit_transform(train_X),
            var.transform(test_X)
        ],
        'filterMethod': [
            chi.fit_transform(train_X, train_y),
            chi.transform(test_X)
        ],
        'wrapperFeature': [
            rfe.fit_transform(train_X, train_y),
            rfe.transform(test_X)
        ],
        'embMethod': [
            svc.fit_transform(train_X, train_y),
            svc.transform(test_X)
        ],
        'featureExtraction': [
            pca.fit_transform(train_X),
            pca.transform(test_X)
        ],
    }


def gridSearch(X, y, params, estimator):
    clf = GridSearchCV(estimator=estimator, param_grid=params, cv=5)
    clf.fit(X, y)
    return clf, clf.best_estimator_, clf.best_score_, \
           clf.cv_results_.get('params'), clf.cv_results_.get('mean_test_score'), \
           clf.cv_results_.get('rank_test_score')


def plotCV(title, params, scores, bestIdx, dimReduction):
    X = []
    for param in params:  # [{'algorithm': 'auto', 'n_neighbors': 1}, {'algorithm': 'auto', 'n_neighbors': 2}]
        x = ''
        for item in param:
            x += f'{item}-{param[item]}'
        X.append(x)
    plt.figure(figsize=(30, 20))
    plt.title(f'{title}\nBest Params: {X[bestIdx]}\n'
              f'Best Score: {scores[bestIdx]}\nDimension Reduction Method: {dimReduction}')
    plt.xlabel('Parameters')
    plt.ylabel('Scores')
    plt.xticks(rotation=45)
    plt.plot(X, scores, marker='o')
    plt.savefig(f'{title}_{dimReduction}.png')
    plt.clf()


if __name__ == '__main__':

    f = open('./log_bi.txt', 'w', encoding='utf-8')

    # A) Data preprocessing
    # simplify to three binary classification problems
    pairs = createPairs(df)

    # B) Model fitting
    models = {
        KNeighborsClassifier(): {
            'n_neighbors': [1, 2, 3, 4, 5],
            'algorithm': ['ball_tree', 'kd_tree', 'brute']
        },
        DecisionTreeClassifier(): {
            'max_depth': [2, 4, 6, 8, 10, 12, 14],
            'max_features': ['auto', 'log2', 'sqrt']
        },
        SVC(): {
            'C': [.01, .1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly']
        },
        RandomForestClassifier(): {
            'n_estimators': [100, 150, 200, 250, 300],
            'max_features': ['float', 'log2', 'sqrt']
        },
        MLPClassifier(max_iter=1000): {
            'activation': ['relu', 'tanh', 'logistic'],
            'learning_rate': ['constant', 'invscaling', 'adaptive']
        },
        AdaBoostClassifier(): {
            'n_estimators': [50, 100, 150, 200, 300],
            'learning_rate': [.1, .25, .5, .9, 1]
        }
    }

    for pair in pairs:
        print('============================\n|            %s            |\n============================' % str(pair))
        f.write('===========================\n|            %s            |\n===========================\n' % str(pair))
        train_X, train_y, test_X, test_y = pairs[pair]

        dimReduc = dimensionReduction(train_X=train_X, test_X=test_X, train_y=train_y, dt=str(pair))

        for model in models:
            print('「MODEL」%s' % str(model))
            f.write('「MODEL」%s\n' % str(model))

            for method in dimReduc:
                print('Dim Reduction: %s' % str(method))
                f.write('Dim Reduction: %s\n' % str(method))
                trainXnew, testXnew = dimReduc[method]
                grid_search = gridSearch(trainXnew, train_y, models[model], model)
                print(f'Best Estimator: {grid_search[1]}, '
                      f'Best Parameters: {grid_search[3][grid_search[5].tolist().index(1)]}')
                f.write(f'Best Estimator: {grid_search[1]}, \n'
                        f'Best Parameters: {grid_search[3][grid_search[5].tolist().index(1)]}\n')
                plotCV(f'{pair}-{str(model)}\n', grid_search[3], grid_search[4], grid_search[5].tolist().index(1), method)

                classifier = grid_search[0]
                testScore = classifier.score(testXnew, test_y)
                print(f'Test Score : {testScore}')
                f.write(f'Test Score : {testScore}\n')
                print()
        print('------------------------------------\n')
        f.write('------------------------------------\n')
        f.close()
