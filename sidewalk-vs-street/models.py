from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from preprocess_data import read_all_stream_files_in_dir


def model_cross_val_stats(X, y):
    lgr_clf = LogisticRegression()
    print('Logistic Regression Classifier CV:')
    print(cross_val_score(lgr_clf, X, y, cv=5))
    print('.....')

    svc_clf = LinearSVC(random_state=0)
    print('LinearSVC Classifier CV:')
    print(cross_val_score(svc_clf, X, y, cv=5))
    print('.....')

    knn_clf = KNeighborsClassifier()
    print('KNN Classifier CV:')
    print(cross_val_score(knn_clf, X, y, cv=5))
    print('.....')

    gnb_clf = GaussianNB()
    print('Gaussian Naive Bayes Classifier CV:')
    print(cross_val_score(gnb_clf, X, y, cv=5))
    print('.....')

    gbc_clf = GradientBoostingClassifier()
    print('Gradient Boosting Classifier CV:')
    print(cross_val_score(gbc_clf, X, y, cv=5))
    print('.....')

    rfc_clf = RandomForestClassifier()
    print('Random Forest Classifier CV:')
    print(cross_val_score(rfc_clf, X, y, cv=5))
    print('.....')


if __name__ == '__main__':
    full_quantized_df = read_all_stream_files_in_dir('IMU_Streams', window_size=150)
    shuffled_data = full_quantized_df.sample(frac=1).reset_index(drop=True)
    X = shuffled_data.iloc[:, :-1]
    y = shuffled_data.iloc[:, -1]

    model_cross_val_stats(X, y)

    print('Done!')
