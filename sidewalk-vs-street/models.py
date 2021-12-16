from numpy.core.numeric import full
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, precision_score, recall_score, PrecisionRecallDisplay, f1_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from preprocess_data import read_all_stream_files_in_dir, shuffle_and_split
from datetime import datetime
import os
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import pandas as pd
import lightgbm as lgb
import xgboost as xgb



if not os.path.isdir("sidewalk-vs-street/imu_classifier_results"):
    os.mkdir("sidewalk-vs-street/imu_classifier_results")

def run_all_model_cross_val_stats(X, y, max_depth=None, max_features='auto', n_estimators=100, svc_kernel='rbf'):
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    print(date_time)
    print("training settings: ")
    print("max depth trees: ", max_depth)
    print("max features trees: ", max_features)
    print("N_estimators trees: ", n_estimators)
    print("svc kernel: ", kernel)
    res_file =  open("sidewalk-vs-street/imu_classifier_results/street_classifier_{}.txt".format(date_time), mode='w')
    results = dict()
    
    val_score, train_score, scores = run_logistic_regression(X, y)
    res_file.write("Logistic Regression Classifier CV: val_score: {}, train_scores: {} \n".format(val_score, train_score))
    results['Logistic Regression Classifier CV:'] = scores
    print(scores)
    print('.....')
    

    val_score, train_score, svc_res = run_svc(X, y, kernel=kernel)
    res_file.write("SVC Classifier with rbf kernel CV: val_score: {}, train_scores: {} \n".format(val_score, train_score))
    print(svc_res)
    results['SVC Classifier with rbf kernel CV:'] = svc_res
    print('.....')

    val_score, train_score, knn_res = run_knn(X, y)
    res_file.write("KNN Classifier CV: val_score: {}, train_scores: {} \n".format(val_score, train_score) )
    print(knn_res)
    results['KNN Classifier CV:'] = knn_res
    print('.....')

    val_score, train_score, gnb_res = run_gaussian_naive_bayes(X, y)
    res_file.write("Gaussian Naive Bayes Classifier CV: val_score: {}, train_scores: {} \n".format(val_score, train_score) )
    print(gnb_res)
    results['Gaussian Naive Bayes Classifier CV:'] = gnb_res
    print('.....')

    val_score, train_score, gbc_res = run_gradient_boosted(X, y, n_estimators=n_estimators)
    res_file.write("Gradient Boosting Classifier CV: val_score: {}, train_scores: {} \n".format(val_score, train_score) )
    print(gbc_res)
    results['Gradient Boosting Classifier CV:'] = gbc_res
    print('.....')

    val_score, train_score, rfc_res = run_random_forest(X, y, n_estimators, max_depth, max_features)
    res_file.write("Random Forest Classifier CV: val_score: {}, train_scores: {} \n".format(val_score, train_score) )
    print(rfc_res)
    results['Random Forest Classifier CV:'] = rfc_res
    print('.....')

    res_file.close()    

def run_logistic_regression(X, y):
    lgr_clf = LogisticRegression()
    print('Logistic Regression Classifier CV:')
    lgr_res = cross_validate(lgr_clf, X, y, cv=5, return_train_score=True)
    val_score = lgr_res['test_score'].mean()
    train_scores = lgr_res['train_score'].mean()
    print(lgr_res)
    print('.....')
    return val_score, train_scores, lgr_res

def run_svc(X,y, kernel='rbf'):
    svc_clf = SVC(random_state=0, kernel=kernel)
    print('SVC Classifier with rbf kernel CV:')
    svc_res = cross_validate(svc_clf, X, y, cv=5, return_train_score=True)
    val_score = svc_res['test_score'].mean()
    train_scores = svc_res['train_score'].mean()
    return val_score, train_scores, svc_res

def run_knn(X, y):
    knn_clf = KNeighborsClassifier()
    print('KNN Classifier CV:')
    knn_res = cross_validate(knn_clf, X, y, cv=5, return_train_score=True)
    val_score = knn_res['test_score'].mean()
    train_scores = knn_res['train_score'].mean()
    return val_score, train_scores, knn_res

def run_gaussian_naive_bayes(X, y):
    gnb_clf = GaussianNB()
    print('Gaussian Naive Bayes Classifier CV:')
    gnb_res = cross_validate(gnb_clf, X, y, cv=5, return_train_score=True)
    val_score = gnb_res['test_score'].mean()
    train_scores = gnb_res['train_score'].mean()
    return val_score, train_scores, gnb_res

def run_gradient_boosted(X, y, n_estimators=100):
    gbc_clf = GradientBoostingClassifier(n_estimators=n_estimators)
    print('Gradient Boosting Classifier CV:')
    gbc_res = cross_validate(gbc_clf, X, y, cv=5, return_train_score=True)
    val_score = gbc_res['test_score'].mean()
    train_scores =  gbc_res['train_score'].mean()
    return val_score, train_scores, gbc_res

def run_random_forest(X, y, n_estimators, max_depth, max_features):
    rfc_clf = RandomForestClassifier(max_depth=max_depth, max_features=max_features, n_estimators=n_estimators)
    print('Random Forest Classifier CV:')
    rfc_res = cross_validate(rfc_clf, X, y, cv=5, return_train_score=True)
    val_score = rfc_res['test_score'].mean()
    train_score = rfc_res['train_score'].mean()
    return val_score, train_score, rfc_res

def parameter_tuning_rf(X_train, y_train, X_test, y_test):
    print('entering param tuning')
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    params = {'max_depth':[2, 3, 4, 5, 6], 
    'max_features': ['auto', 'sqrt', 'log2'], 
    'n_estimators': [50, 100, 150, 200, 250, 300, 350, 500]}
    rfc = RandomForestClassifier()
    clf = GridSearchCV(rfc, param_grid=params, n_jobs=-1)
    print('initializing params search...')
    clf.fit(X_train, y_train)
    print('fitted best model')
    
    best_rf = clf.best_estimator_
    best_score = clf.best_score_
    best_params = clf.best_params_
    y_pred = clf.predict(X_test)
    print("pre smoothing")
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("f1: ", f1)
    print("precision ", prec)
    print("recall ", recall)
    #output smoothing (by most common of last 5 outputs)
    for i in range(SMOOTH_STEP, len(y_pred), SMOOTH_STEP):
        slice = y_pred[i-SMOOTH_STEP:i]
        y_pred[i-SMOOTH_STEP:i] = majority_vote(slice)    

    print('predictions made')
    #confusion_matrix = 
    #confusion_matrix = confusion_matrix(y_test, y_pred)
    disp1 = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=['sidewalk', 'street'])
    disp1.plot()
    print("post smoothing")
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("f1: ", f1)
    print("precision ", prec)
    print("recall ", recall)
    prec, recall, _ = precision_recall_curve(y_test, y_pred)
    disp2 = PrecisionRecallDisplay(precision=prec, recall=recall)
    disp2.plot()
    plt.show()
    
    with open("sidewalk-vs-street/imu_classifier_results/random_forest_search_{}.txt".format(date_time), 'w') as res_file:
        res_file.write("Random Forest Classifier Grid Search: \n")
        res_file.write("best val score: " + str(best_score)+'\n')
        res_file.write("best parameters: " + str(best_params)+ "\n")
        res_file.write("test f1 score: "+ str(f1)+'\n')
        res_file.write('test precison: '+str(prec)+'\n')
        res_file.write('test recall: '+str(recall)+'\n')
    print('done! results written to file')  

def compare_by_sublabel(y_pred, y_test, sublabels, title):
    correct_results = defaultdict(int)
    wrong_results = defaultdict(int)
    for i in range(len(sublabels)):
        if y_pred[i] == y_test[i]:
            correct_results[sublabels[i]] +=1
        else:
            wrong_results[sublabels[i]] +=1
    print("correct: ", correct_results)
    print("wrong: ", wrong_results)
    print("total correct: ", sum(correct_results.values()))
    print("total incorrect: ", sum(wrong_results.values()))
    '''fig, ax = plt.subplots(1, 2)
    ax[0].bar(x=correct_results.keys(), height=correct_results.values(), color='g')
    ax[1].bar(x=wrong_results.keys(), height=wrong_results.values(), color='r')
    ax[0].set_ylabel("number of samples")
    ax[0].set_xlabel("sublabel identity")
    ax[1].set_ylabel("number of samples")
    ax[1].set_xlabel("sublabel identity")
    ax[0].set_title("correctly classified samples from each subclass")
    ax[1].set_title("wrongly classified samples from each subclass")
    #plt.bar(x=results.keys(), height=results.values(), color=['red', 'blue', 'green', 'purple', 'orange'])
    plt.title(title)
    #ax.set_xticks(correct_results.keys())
    #ax.legends(labels=['correct', 'misclassified'])
    plt.savefig(title+'.png')
    plt.show()'''
    with open("comparison_by_sublabel_{}".format(title), mode='w') as file:
        file.write("correct results\n")
        for key, val in zip(correct_results.keys(), correct_results.values()):
            file.write(key+"--->"+str(val)+'\n')
        file.write("wrong results\n")
        for key, val in zip(wrong_results.keys(), wrong_results.values()):
            file.write(key+"--->"+str(val)+'\n')


def majority_vote(lst):
    data = Counter(lst)
    return max(lst, key=data.get)

def run_classifier_rf(X_train, y_train, X_test, y_test, max_depth, max_features, n_estimators, SMOOTH_STEP):
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    rfc_clf = RandomForestClassifier()
    rfc_clf.fit(X_train, y_train)
    y_pred = rfc_clf.predict(X_test)
    disp1 = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=['sidewalk', 'street'])
    disp1.plot()
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    #output smoothing (by most common of last 5 outputs)
    plt.close()
    y_pred_smooth = np.zeros_like(y_pred)
    for i in range(SMOOTH_STEP, len(y_pred), SMOOTH_STEP):
        slice = y_pred[i-SMOOTH_STEP:i]
        y_pred_smooth[i-SMOOTH_STEP:i] = majority_vote(slice)
    f1_smooth = f1_score(y_test, y_pred_smooth)
    prec_smooth = precision_score(y_test, y_pred_smooth)
    recall_smooth = recall_score(y_test, y_pred_smooth)
    disp2 = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_smooth), display_labels=['sidewalk', 'street'])
    disp2.plot()
    with open("sidewalk-vs-street/imu_classifier_results/random_forest_final_{}.txt".format(date_time), mode='w') as rf_file:
        rf_file.write("Random Forest, settings: \n max depth {} max features {} num trees {} \n".format(max_depth, max_features, n_estimators))
        rf_file.write("F1 score normal {} F1 score smoothed {} \n".format(f1, f1_smooth))
        rf_file.write("precision: {} smoothed precision: {}\n".format(prec, prec_smooth))
        rf_file.write("recall {}, recall smoothed {} \n".format(recall, recall_smooth))
    print('Random Forest Classifier Results: ')
    print("F1 score {}, smoothed F1 score {}".format(f1, f1_smooth))
    print("precision {} smoothed precision {}".format(prec, prec_smooth))
    print("recall {} recall smooth {}".format(recall, recall_smooth))
    print("done!")
    plt.close()
    return y_pred, y_pred_smooth


def run_lgbm(X_train, y_train, X_test, Y_test):
    print('light gradient boosted machine')
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    train = lgb.Dataset(X_train, label=y_train)
    params = {'objective':'binary'}
    print('starting cross validate')
    eval_hist = lgb.cv(params, train)
    print(eval_hist)
    with open("sidewalk-vs-street/imu_classifier_results/lgb_results_{}.txt".format(date_time), mode='w') as file:
        file.write(eval_hist)
    print("done, results written to file")
    
    

if __name__ == '__main__':
     #HYPERPARAMETERS
    mode='running_window'
    test_size = 0.80
    shuffle = True
    WINDOW_SIZE = 75
    kernel = 'rbf'
    n_estimators = 100
    max_features = 'auto'
    max_depth = None
    SMOOTH_STEP = 5
    #train, test = read_all_stream_files_in_dir('IMU_Streams', window_size=WINDOW_SIZE, mode=mode)
    #print("dataset generated")

    

    #print('saved to csv')
    train, test = read_all_stream_files_in_dir("IMU_Streams", test_size=test_size, shuffle=shuffle, window_size=WINDOW_SIZE, mode=mode)
    train.to_csv("IMU_Streams/train_samples_{}_shuffled_80.csv".format(mode))
    test.to_csv("IMU_Streams/test_samples_{}_shuffled_80.csv".format(mode))
    
    #train, test = shuffle_and_split(all_samples, test_size=0.20, shuffle=True)
    #load train and test files:
    train = pd.read_csv("IMU_Streams/train_samples_{}_shuffled_80.csv".format(mode))
    test = pd.read_csv("IMU_Streams/test_samples_{}_shuffled_80.csv".format(mode))
    print('loaded data from csv')
    
    print("number of training/val samples: ", train.shape[0])
    print("number of test samples: ", test.shape[0])
    
    X_train = train.iloc[:, :-2]
    sublabels = train.iloc[:,-2]
    y_train = train.iloc[:, -1]
    X_test = test.iloc[:,:-2]
    sublabels_test = test.iloc[:,-2]
    y_test = test.iloc[:, -1]
    print('number of sublabels in train and test: {} {}'.format(len(sublabels), len(sublabels_test)))
    #parameter_tuning_rf(X_train, y_train, X_test, y_test)
    #run_lgbm(X_train, y_train, X_test, y_test)
    #all_data = pd.concat((X_train, X_test), axis=0)
    #all_data_y = pd.concat
    #((y_train, y_test), axis=0)
    #run_all_model_cross_val_stats(X_train, y_train, max_depth=max_depth, max_features=max_features, n_estimators=n_estimators)
    #val_score, train_score, full_results = run_random_forest(X_train, y_train, n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
    #print("random forest")
    #print("val score: ", val_score)
    #print("train score ", train_score)
    #print("full scores: ", full_results)
    y_pred, y_pred_smooth = run_classifier_rf(X_train, y_train, X_test, y_test, max_depth=max_depth, 
    max_features=max_features, n_estimators=n_estimators, SMOOTH_STEP=SMOOTH_STEP)
    print(len(y_test))
    print(len(y_pred))
    y_pred = np.array(y_pred).reshape(-1,1)
    y_pred_smooth = np.array(y_pred_smooth).reshape(-1,1)
    y_test = np.array(y_test).reshape(-1,1)
    np.savetxt('y_pred.csv', np.hstack((y_pred, y_test)), delimiter=',')
    np.savetxt('y_pred_smooth.csv', np.hstack((y_pred_smooth, y_test)), delimiter=',')
    compare_by_sublabel(y_pred, y_test, sublabels_test, title='sublabel_comparison_rf_running_win_reg')
    plt.close()
    compare_by_sublabel(y_pred_smooth, y_test, sublabels_test, title='sublabel_comparison_rf_run_win_smoothed')
    
    '''
    step_tests = np.array([3, 5, 8, 10])
    print("starting experiments")
    for step in step_tests:
        print("smoothing steps: {}".format(step))
        title = 'classifications_by_sublabel_'
        y_pred, y_pred_smooth = run_classifier_rf(X_train, y_train, X_test, y_test, max_depth, max_features, n_estimators, step)
        compare_by_sublabel(y_pred, y_test, sublabels_test, title=title)
        compare_by_sublabel(y_pred_smooth, y_test, sublabels_test, title=title+"smooth_level_{}".format(step))
    print("done! ")'''
