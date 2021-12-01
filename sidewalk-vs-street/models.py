from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from preprocess_data import read_all_stream_files_in_dir
from datetime import datetime
import os


if not os.path.isdir("sidewalk-vs-street/imu_classifier_results"):
    os.mkdir("sidewalk-vs-street/imu_classifier_results")

def model_cross_val_stats(X, y):
    results = dict()
    lgr_clf = LogisticRegression()
    print('Logistic Regression Classifier CV:')
    lgr_res = cross_val_score(lgr_clf, X, y, cv=5)
    results['Logistic Regression Classifier CV:'] = lgr_res
    print(lgr_res)
    print('.....')

    svc_clf = LinearSVC(random_state=0)
    print('LinearSVC Classifier CV:')
    svc_res = cross_val_score(svc_clf, X, y, cv=5)
    print(svc_res)
    results['LinearSVC Classifier CV:'] = svc_res
    print('.....')

    knn_clf = KNeighborsClassifier()
    print('KNN Classifier CV:')
    knn_res =cross_val_score(knn_clf, X, y, cv=5)
    print(knn_res)
    results['KNN Classifier CV:'] = knn_res
    print('.....')

    gnb_clf = GaussianNB()
    print('Gaussian Naive Bayes Classifier CV:')
    gnb_res = cross_val_score(gnb_clf, X, y, cv=5)
    print(gnb_res)
    results['Gaussian Naive Bayes Classifier CV:'] = gnb_res
    print('.....')

    gbc_clf = GradientBoostingClassifier()
    print('Gradient Boosting Classifier CV:')
    gbc_res = cross_val_score(gbc_clf, X, y, cv=5)
    print(gbc_res)
    results['Gradient Boosting Classifier CV:'] = gbc_res
    print('.....')

    rfc_clf = RandomForestClassifier()
    print('Random Forest Classifier CV:')
    rfc_res = cross_val_score(rfc_clf, X, y, cv=5)
    print(rfc_res)
    results['Random Forest Classifier CV:'] = rfc_res
    print('.....')
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    print(date_time)
    with open("sidewalk-vs-street/imu_classifier_results/street_classifier_{}.txt".format(date_time), mode='w') as res_file:
        res_file.writelines(results)


if __name__ == '__main__':
    full_quantized_df = read_all_stream_files_in_dir('IMU_Streams', window_size=150)
    shuffled_data = full_quantized_df.sample(frac=1).reset_index(drop=True)
    X = shuffled_data.iloc[:, :-1]
    y = shuffled_data.iloc[:, -1]

    model_cross_val_stats(X, y)

    print('Done!')
