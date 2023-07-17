import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from binary import dimensionReduction, gridSearch, plotCV


# Loading the dataset
cols = ['lettr', 'x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar', 'y-bar', 'x2bar',
        'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx']
df = pd.read_csv('./letter-recognition.data.txt', encoding='utf-8', header=None, names=cols)

# Splitting the dataset into train and test sets
X = df.iloc[:, 1:]
y = df.iloc[:, 0]
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=42)

models = {
        KNeighborsClassifier(): {
            'n_neighbors': [10, 12, 14, 16, 18],
            'algorithm': ['ball_tree', 'kd_tree', 'brute']
        },
        DecisionTreeClassifier(): {
            'max_depth': [20, 25, 30, 35, 40],
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


dimReduc = dimensionReduction(train_X=train_X, test_X=test_X, train_y=train_y)

f = open('./log_multi.txt', 'w', encoding='utf-8')
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
        plotCV(f'{str(model)}\n', grid_search[3], grid_search[4], grid_search[5].tolist().index(1), method)

        classifier = grid_search[0]
        testScore = classifier.score(testXnew, test_y)
        print(f'Test Score : {testScore}')
        f.write(f'Test Score : {testScore}\n')
        print()
print('------------------------------------\n')
f.write('------------------------------------\n')
f.close()

# # Training the model
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)
#
# # Making predictions on the test set
# y_pred = rf.predict(X_test)
#
# # Evaluating the model's accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print('Accuracy:', accuracy)
