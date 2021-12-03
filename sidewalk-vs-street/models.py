from numpy.core.numeric import full
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from preprocess_data import read_all_stream_files_in_dir, shuffle_and_split
from datetime import datetime
import os

WINDOW_SIZE = 75

if not os.path.isdir("sidewalk-vs-street/imu_classifier_results"):
    os.mkdir("sidewalk-vs-street/imu_classifier_results")

def model_cross_val_stats(X, y):
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    print(date_time)
    res_file =  open("sidewalk-vs-street/imu_classifier_results/street_classifier_{}.txt".format(date_time), mode='w')
    results = dict()
    lgr_clf = LogisticRegression()
    scores = dict()
    print('Logistic Regression Classifier CV:')
    lgr_res = cross_validate(lgr_clf, X, y, cv=5)
    val_score = lgr_res['test_score'].mean()
    train_scores = lgr_res['train_score']
    res_file.write("Logistic Regression Classifier CV: val_score: {}, train_scores: {}".format(val_score, train_scores))
    scores['val_score'] = val_score
    scores['train_scores'] = train_scores
    results['Logistic Regression Classifier CV:'] = scores
    print(lgr_res)
    print('.....')

    svc_clf = SVC(random_state=0, kernel='rbf')
    print('SVC Classifier with rbf kernel CV:')
    svc_res = cross_validate(svc_clf, X, y, cv=5)
    val_score = svc_res['test_score'].mean()
    train_scores = svc_res['train_score']
    res_file.write("Logistic Regression Classifier CV: val_score: {}, train_scores: {}".format(val_score, train_scores) )
    print(svc_res)
    results['SVC Classifier with rbf kernel CV:'] = svc_res
    print('.....')

    knn_clf = KNeighborsClassifier()
    print('KNN Classifier CV:')
    knn_res = cross_validate(knn_clf, X, y, cv=5)
    val_score = knn_res['test_score'].mean()
    train_scores = knn_res['train_score']
    res_file.write("Logistic Regression Classifier CV: val_score: {}, train_scores: {}".format(val_score, train_scores) )
    print(knn_res)
    results['KNN Classifier CV:'] = knn_res
    print('.....')

    gnb_clf = GaussianNB()
    print('Gaussian Naive Bayes Classifier CV:')
    gnb_res = cross_validate(gnb_clf, X, y, cv=5)
    val_score = gnb_res['test_score'].mean()
    train_scores = gnb_res['train_score']
    res_file.write("Logistic Regression Classifier CV: val_score: {}, train_scores: {}".format(val_score, train_scores) )
    print(gnb_res)
    results['Gaussian Naive Bayes Classifier CV:'] = gnb_res
    print('.....')

    gbc_clf = GradientBoostingClassifier(n_estimators=500)
    print('Gradient Boosting Classifier CV:')
    gbc_res = cross_validate(gbc_clf, X, y, cv=5)
    val_score = gbc_res['test_score'].mean()
    train_scores =  gbc_res['train_score']
    res_file.write("Logistic Regression Classifier CV: val_score: {}, train_scores: {}".format(val_score, train_scores) )
    print(gbc_res)
    results['Gradient Boosting Classifier CV:'] = gbc_res
    print('.....')

    rfc_clf = RandomForestClassifier(n_estimators=500)
    print('Random Forest Classifier CV:')
    rfc_res = cross_validate(rfc_clf, X, y, cv=5)
    val_score = rfc_res['test_score'].mean()
    train_scores = rfc_res['train_score']
    res_file.write("Logistic Regression Classifier CV: val_score: {}, train_scores: {}".format(val_score, train_scores) )
    print(rfc_res)
    results['Random Forest Classifier CV:'] = rfc_res
    print('.....')

    res_file.close()
    
    #with open("sidewalk-vs-street/imu_classifier_results/street_classifier_{}.txt".format(date_time), mode='w') as res_file:
    #    for key, value in results.items():
    #        res_file.write(key +  "--->" + str(value) + "\n")


if __name__ == '__main__':
    full_quantized_df = read_all_stream_files_in_dir('IMU_Streams', window_size=WINDOW_SIZE)

    #shuffled_train_data = full_quantized_df.sample(frac=0.85)
    train, test = shuffle_and_split(full_quantized_df, test_size=0.001)
    #test_data = full_quantized_df.drop(shuffled_train_data.index).reset_index(drop=True)
    print("number of training/val samples: ", train.shape[0])
    print("number of test samples: ", test.shape[0])
    X = train.iloc[:, :-1]
    y = train.iloc[:, -1]

    model_cross_val_stats(X, y)

    print('Done!')
