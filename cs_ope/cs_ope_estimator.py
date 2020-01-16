import numpy as np
import pandas as pd

from kernel_regression import KernelRegression

from sklearn import linear_model

from sklearn.neighbors import KernelDensity
from densratio import densratio


def nadaraya_watson(target, X, Y, bandwidth):
    Z = np.mean((X - target)**2, axis=1)
    K = np.exp(-Z/bandwidth)
    f = np.mean(K*Y)/np.mean(K)
    return f

def ipw(Y_hst, X_hst, X_evl, classes, pi_evaluation, pi_behavior=None, A_hst=None, p=None, q=None):
    N_hst = len(Y_hst)
    
    if p is None:
        kde_train = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X_hst)
        p = np.exp(kde_train.score_samples(X_hst))
        
    if q is None:
        kde_test = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X_evl)
        q = np.exp(kde_test.score_samples(X_hst))
        
    if pi_behavior is None:
        pi_behavior = np.zeros(shape=(N_hst, len(classes)))
        for c in classes:
            pi_behavior_list = []
            for i in range(N_hst):
                pi_behavior_list.append(nadaraya_watson(X_hst[i], X_hst, A_hst[:,c], 0.01))
            pi_behavior_list = np.array(pi_behavior_list)
            pi_behavior[:, c] = pi_behavior_list
        print(pi_behavior)
    
    r = q/p
    w = pi_evaluation/pi_behavior
    
    r = np.array([r for c in range(len(classes))]).T
    
    return np.sum(Y_hst*w*r)/len(Y_hst)

def dm(Y_hst, X_hst, X_evl, pi_evaluation, classes):
    N_evl = len(X_evl)
    
    f_matrix = np.zeros(shape=(N_evl, len(classes)))
    for c in classes:
        f_list = []
        for i in range(N_evl):
            f_list.append(nadaraya_watson(X_evl[i], X_hst, Y_hst[:,c], 0.01))
        f_list = np.array(f_list)
        f_matrix[:, c] = f_list
        
    return np.sum(f_matrix*pi_evaluation)/N_evl

def dr(Y_hst, A_hst, X_hst, X_evl, pi_evaluation, pi_evaluation_test, classes, folds=2, pi_behavior=None, p=None, q=None, w=None, method='Lasso'):
    N_hst = len(Y_hst)
    N_evl = len(X_evl)

    w = pi_evaluation/pi_behavior

    f_hst_matrix = np.zeros(shape=(len(X_hst), len(classes)))
    f_evl_matrix = np.zeros(shape=(len(X_evl), len(classes)))
    w_matrix = np.zeros(shape=(len(w), len(classes)))

    for c in classes:
        clf, x_ker_train, x_ker_test = KernelRegression(X_hst, Y_hst[:, c], X_evl, algorithm=method)
        clf.fit(x_ker_train, Y_hst[:, c])
        f_hst_matrix[:, c] = clf.predict(x_ker_train)
        f_evl_matrix[:, c] = clf.predict(x_ker_test)

        clf, x_ker_train, x_ker_test = KernelRegression(X_hst, w[:, c], X_evl, algorithm=method)
        clf.fit(x_ker_train, w[:, c])
        w_matrix[:, c] = clf.predict(x_ker_train)

    densratio_obj = densratio(X_evl, X_hst)
    r = densratio_obj.compute_density_ratio(X_hst)
    r = np.array([r for c in classes]).T

    theta = np.sum((Y_hst-f_hst_matrix)*w_matrix*r)/N_hst + np.sum(f_evl_matrix*pi_evaluation_test)/N_evl
    
    return theta

def dml(Y_hst, A_hst, X_hst, X_evl, pi_evaluation, pi_evaluation_test, classes, folds=2, pi_behavior=None, p=None, q=None, w_est=False, method='Lasso'):
    N_hst = len(Y_hst)
    N_evl = len(X_evl)
    
    theta_list = []
    
    cv_hst_fold = np.arange(folds) 
    cv_hst_split0 = np.floor(np.arange(N_hst)*folds/N_hst)
    cv_hst_index = cv_hst_split0[np.random.permutation(N_hst)]
    
    cv_evl_fold = np.arange(folds) 
    cv_evl_split0 = np.floor(np.arange(N_evl)*folds/N_evl)
    cv_evl_index = cv_evl_split0[np.random.permutation(N_evl)]

    x_hst_cv = []
    x_evl_cv = []
    y_cv = []
    p_bhv_cv = []
    p_evl_cv = []
    p_evl_test_cv = []
    
    for k in cv_hst_fold:
        x_hst_cv.append(X_hst[cv_hst_index==k])
        x_evl_cv.append(X_evl[cv_evl_index==k])
        y_cv.append(Y_hst[cv_hst_index==k])
        p_bhv_cv.append(pi_behavior[cv_hst_index==k])
        p_evl_cv.append(pi_evaluation[cv_hst_index==k])
        p_evl_test_cv.append(pi_evaluation_test[cv_evl_index==k])

    for k in range(folds):
        #print(h0_cv[0])
        # calculate the h vectors for training and test
        count = 0
        for j in range(folds):
            if j == k:
                x_hst_te = x_hst_cv[j]
                x_evl_te = x_evl_cv[j]
                y_te = y_cv[j]
                p_bhv_te = p_bhv_cv[j]
                p_evl_te = p_evl_cv[j]
                p_evl_test_te = p_evl_test_cv[j]
            else:
                if count == 0:
                    x_hst_tr = x_hst_cv[j]
                    x_evl_tr = x_evl_cv[j]
                    y_tr = y_cv[j]
                    p_bhv_tr = p_bhv_cv[j]
                    p_evl_tr = p_evl_cv[j]
                    p_evl_test_tr = p_evl_test_cv[j]
                    count += 1
                else:
                    x_hst_tr = np.append(x_hst_tr, x_hst_cv[j], axis=0)
                    x_evl_tr = np.append(x_evl_tr, x_evl_cv[j], axis=0)
                    y_tr = np.append(y_tr, y_cv[j], axis=0)
                    p_bhv_tr = np.append(p_bhv_tr, p_bhv_cv[j], axis=0)
                    p_evl_tr = np.append(p_evl_tr, p_evl_cv[j], axis=0)
                    p_evl_test_tr = np.append(p_evl_test_tr, p_evl_test_cv[j], axis=0)
        
        w = p_evl_tr/p_bhv_tr
        
        f_hst_matrix = np.zeros(shape=(len(x_hst_tr), len(classes)))
        f_evl_matrix = np.zeros(shape=(len(x_evl_tr), len(classes)))
        w_matrix = np.zeros(shape=(len(w), len(classes)))
        
        for c in classes:
            clf, x_ker_train, x_ker_test = KernelRegression(x_hst_tr, y_tr[:, c], x_evl_tr, algorithm=method)
            clf.fit(x_ker_train, y_tr[:, c])
            f_hst_matrix[:, c] = clf.predict(x_ker_train)
            f_evl_matrix[:, c] = clf.predict(x_ker_test)
            
            clf, x_ker_train, x_ker_test = KernelRegression(x_hst_tr, w[:, c], x_evl_tr, algorithm=method)
            clf.fit(x_ker_train, w[:, c])
            
            if w_est is False:
                w_matrix[:, c] = clf.predict(x_ker_train)
            else:
                w_matrix[:, c] = w[:, c]
        
        densratio_obj = densratio(x_evl_tr, x_hst_tr)
        r = densratio_obj.compute_density_ratio(x_hst_tr)
        r = np.array([r for c in classes]).T
        
        theta = np.sum((y_tr-f_hst_matrix)*w_matrix*r)/N_hst + np.sum(f_evl_matrix*p_evl_test_tr)/N_evl
        theta_list.append(theta)
    
    return np.mean(theta_list)
