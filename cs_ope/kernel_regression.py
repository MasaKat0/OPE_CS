import numpy as np
from sklearn import linear_model

import numpy as np


def dist(x, T=None, num_basis=False):

    (d,n) = x.shape

    # check input argument

    if num_basis is False:
        num_basis = 100000

    idx = np.random.permutation(n)[0:num_basis]
    C = x[:, idx]

    # calculate the squared distances
    XC_dist = CalcDistanceSquared(x, C)
    TC_dist = CalcDistanceSquared(T, C)
    CC_dist = CalcDistanceSquared(C, C)

    return XC_dist, TC_dist, CC_dist, n, num_basis

def KernelRegression(x_train, t_train, x_test, folds=5, num_basis=False, sigma_list=None, lda_list=None, algorithm='Ridge', logit=False):
    x_train, x_test = x_train.T, x_test.T
    t_train = t_train
    XC_dist, TC_dist, CC_dist, n, num_basis = dist(x_train, x_test, num_basis)
    # setup the cross validation
    cv_fold = np.arange(folds) # normal range behaves strange with == sign
    cv_split0 = np.floor(np.arange(n)*folds/n)
    cv_index = cv_split0[np.random.permutation(n)]
    # set the sigma list and lambda list
    if sigma_list==None:
        sigma_list = np.array([0.001, 0.01, 0.1, 1, 10, 100])
    if lda_list==None:
        lda_list = np.array([0.001, 0.01, 0.1, 1., 10., 100])

    score_cv = np.zeros((len(sigma_list), len(lda_list)))

    for sigma_idx, sigma in enumerate(sigma_list):

        # pre-sum to speed up calculation
        h_cv = []
        t_cv = []
        for k in cv_fold:
            h_cv.append(np.exp(-XC_dist[:, cv_index==k]/(2*sigma**2)))
            t_cv.append(t_train[cv_index==k])

        for k in range(folds):
            #print(h0_cv[0])
            # calculate the h vectors for training and test
            count = 0
            for j in range(folds):
                if j == k:
                    hte = h_cv[j].T
                    tte = t_cv[j]
                else:
                    if count == 0:
                        htr = h_cv[j].T
                        ttr = t_cv[j]
                        count += 1
                    else:
                        htr = np.append(htr, h_cv[j].T, axis=0)
                        ttr = np.append(ttr, t_cv[j], axis=0)

            one = np.ones((len(htr),1))
            htr = np.concatenate([htr, one], axis=1)
            one = np.ones((len(hte),1))
            hte = np.concatenate([hte, one], axis=1)
            for lda_idx, lda in enumerate(lda_list):
                if algorithm == 'Lasso':
                    if logit is True:
                        clf = linear_model.LogisticRegression(penalty='l1', C=lda, solver='saga', multi_class='auto')
                    else:
                        clf = linear_model.Lasso(lda)
                elif algorithm == 'Ridge':
                    if logit is True:
                        clf = linear_model.LogisticRegression(penalty='l2', C=lda, solver='saga', multi_class='auto')
                    else:
                        clf = linear_model.Lasso(lda)
                
                clf.fit(htr, ttr.T)
                pred = clf.predict(hte)
                if logit:
                    score = -np.mean((pred == tte)**2)
                else:
                    score = np.mean((pred - tte)**2)
                """
                if math.isnan(score):
                    code.interact(local=dict(globals(), **locals()))
                """

                score_cv[sigma_idx, lda_idx] = score_cv[sigma_idx, lda_idx] + score

    (sigma_idx_chosen, lda_idx_chosen) = np.unravel_index(np.argmin(score_cv), score_cv.shape)
    sigma_chosen = sigma_list[sigma_idx_chosen]
    lda_chosen = lda_list[lda_idx_chosen]

    x_train = np.exp(-XC_dist/(2*sigma_chosen**2)).T
    x_test = np.exp(-TC_dist/(2*sigma_chosen**2)).T

    one = np.ones((len(x_train),1))
    x_train = np.concatenate([x_train, one], axis=1)
    one = np.ones((len(x_test),1))
    x_test = np.concatenate([x_test, one], axis=1)

    if algorithm == 'Lasso':
        if logit is True:
            clf = linear_model.LogisticRegression(penalty='l1', C=lda_chosen, solver='saga', multi_class='auto')
        else:
            clf = linear_model.Lasso(lda_chosen)
    elif algorithm == 'Ridge':
        if logit is True:
            clf = linear_model.LogisticRegression(penalty='l2', C=lda_chosen, solver='saga', multi_class='auto')
        else:
            clf = linear_model.Ridge(lda_chosen)

    return clf, x_train, x_test

def CalcDistanceSquared(X, C):
    '''
    Calculates the squared distance between X and C.
    XC_dist2 = CalcDistSquared(X, C)
    [XC_dist2]_{ij} = ||X[:, j] - C[:, i]||2
    :param X: dxn: First set of vectors
    :param C: d:nc Second set of vectors
    :return: XC_dist2: The squared distance nc x n
    '''

    Xsum = np.sum(X**2, axis=0).T
    Csum = np.sum(C**2, axis=0)
    XC_dist = Xsum[np.newaxis, :] + Csum[:, np.newaxis] - 2*np.dot(C.T, X)
    return XC_dist