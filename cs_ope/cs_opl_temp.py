import numpy as np
import pandas as pdå
import statsmodels.api as sm

from kernel_regression import KernelRegression
from statsmodels.nonparametric.kernel_regression import KernelReg
from sklearn.neighbors import KernelDensity
from densratio import densratio

from sklearn.kernel_ridge import KernelRidge


class ope_estimators():
    def __init__(self, X, A, Y_matrix, Z, classes, pi_evaluation_train, pi_evaluation_test):
        self.X = X
        self.A = A
        self.Y = Y_matrix
        self.Z = Z
        self.classes = classes
        self.pi_evaluation_train = pi_evaluation_train
        self.pi_evaluation_test = pi_evaluation_test

        self.N_hst, self.dim = X.shape
        self.N_evl = len(Z)

        self.f_hat_kernel = None
        self.bpol_hat_kernel = None
        self.q_hat_kernel = None
        self.p_hat_kernel = None
        self.prepare = False

    def regression(self):        
        f_matrix = np.zeros(shape=(self.N_evl, len(self.classes)))

        if self.f_hat_kernel is None:
            for c in self.classes:
                f_list = []
                perm = np.random.permutation(self.N_hst)
                Y_temp = self.Y[perm[:100]]
                X_temp = self.X[perm[:100]]
                model = KernelReg(Y_temp[:,c], X_temp, var_type='c'*self.dim, reg_type='lc')
                model = KernelReg(self.Y[:,c], self.X, var_type='c'*self.dim, reg_type='lc', bw=model.bw)
                mu, _ = model.fit(self.Z)
                f_matrix[:, c] = mu
            
            self.f_hat_kernel = f_list
            
        return np.sum(f_matrix*self.pi_evaluation_test)/self.N_evl

    def prepare(self, folds=2, method='Lasso'):
        self.r_array = np.zeros((self.N_hst, len(self.classes)))
        self.f_hst_array = np.zeros((self.N_hst, len(self.classes)))
        self.f_evl_array = np.zeros((self.N_evl, len(self.classes)))
        self.bpol_array = np.zeros((self.N_hst, len(self.classes)))
        self.sn_hst_array = np.zeros((self.N_hst, len(self.classes)))
        
        cv_hst_fold = np.arange(folds) 

        cv_hst_split = np.floor(np.arange(self.N_hst)*folds/self.N_hst)
        perm_hst = np.random.permutation(self.N_hst)
        cv_hst_index = cv_hst_split[perm_hst]
        
        cv_evl_split = np.floor(np.arange(self.N_evl)*folds/self.N_evl)
        perm_evl = np.random.permutation(self.N_evl)
        cv_evl_index = cv_evl_split[perm_evl]

        x_cv = []
        a_cv = []
        y_cv = []
        z_cv = []
        p_evl_hst_cv = []
        p_evl_evl_cv = []
        perm_hst_cv = []
        perm_evl_cv = []
        
        for k in cv_hst_fold:
            x_cv.append(self.X[cv_hst_index==k])
            a_cv.append(self.A[cv_hst_index==k])
            y_cv.append(self.Y[cv_hst_index==k])
            z_cv.append(self.Z[cv_evl_index==k])
            p_evl_hst_cv.append(self.pi_evaluation_train[cv_hst_index==k])
            p_evl_evl_cv.append(self.pi_evaluation_test[cv_evl_index==k])
            perm_hst_cv.append(self.perm_hst[cv_hst_index==k])
            perm_evl_cv.append(self.perm_evl[cv_evl_index==k])

        for k in range(folds):
            #print(h0_cv[0])
            # calculate the h vectors for training and test
            count = 0
            for j in range(folds):
                if j != k:
                    if count == 0:
                        x_tr = x_cv[j]
                        a_tr = a_cv[j]
                        y_tr = y_cv[j]
                        z_tr = z_cv[j]
                        p_evl_hst_tr = p_evl_hst_cv[j]
                        p_evl_evl_tr = p_evl_evl_cv[j]
                        perm_hst_tr = perm_hst_cv[j]
                        perm_evl_tr = perm_evl_cv[j]
                        count += 1
                    else:
                        x_tr = np.append(x_tr, x_cv[j], axis=0)
                        a_tr = np.append(a_tr, a_cv[j], axis=0)
                        y_tr = np.append(y_tr, y_cv[j], axis=0)
                        z_tr = np.append(z_tr, z_cv[j], axis=0)
                        p_evl_hst_tr = np.append(p_evl_hst_tr, p_evl_hst_cv[j], axis=0)
                        p_evl_evl_tr = np.append(p_evl_evl_tr, p_evl_evl_cv[j], axis=0)
                        perm_hst_tr = np.append(perm_hst_tr, perm_hst_cv[j], axis=0)
                        perm_evl_tr = np.append(perm_evl_tr, perm_evl_cv[j], axis=0)
                        
            f_hst_matrix = np.zeros(shape=(len(x_tr), len(self.classes)))
            f_evl_matrix = np.zeros(shape=(len(z_tr), len(self.classes)))

            sn_matrix = np.ones(shape=(len(x_tr), len(self.classes)))

            _, a_temp = np.where(a_tr == 1)
            clf, x_ker_train, x_ker_test = KernelRegression(x_tr, a_temp, z_tr, algorithm=method, logit=True)
            clf.fit(x_ker_train, a_temp)
            p_bhv = clf.predict_proba(x_ker_train)
            
            for c in self.classes:
                clf, x_ker_train, x_ker_test = KernelRegression(x_tr, y_tr[:, c], z_tr, algorithm=method, logit=False)
                clf.fit(x_ker_train, y_tr[:, c])
                f_hst_matrix[:, c] = clf.predict(x_ker_train)
                f_evl_matrix[:, c] = clf.predict(x_ker_test)

                sn_matrix[:, c] = np.sum(a_tr[:, c]/p_bhv[:, c])
            
            densratio_obj = densratio(z_tr, x_tr)
            r = densratio_obj.compute_density_ratio(x_tr)
            r = np.array([r for c in self.classes]).T

            self.r_array[perm_hst_tr] = r
            self.f_hst_array[perm_hst_tr] = f_hst_matrix
            self.f_evl_array[perm_hst_tr] = f_evl_matrix
            self.bpol_array[perm_hst_tr] = p_bhv
            self.sn_hst_array[perm_hst_tr] = sn_matrix
        
        self.prepare = True
            
    def denominator(self, self_norm):
        if self_norm:
            if self.bpol_hat_kernel is None:
                pi_behavior = np.zeros(shape=(self.N_hst, len(self.classes)))
                for c in self.classes:
                    perm = np.random.permutation(self.N_hst)
                    A_temp = self.A[perm[:50]]
                    X_temp = self.X[perm[:50]]
                    model = KernelReg(A_temp[:,c], X_temp, var_type='c'*self.dim, reg_type='lc')
                    model = KernelReg(self.A[:,c], self.X, var_type='c'*self.dim, reg_type='lc', bw=model.bw)
                    #model = KernelReg(self.A[:,c], self.X, var_type='c'*self.dim, reg_type='lc')
                    mu, _ = model.fit(self.X)
                    pi_behavior[:, c] = mu
                    
                self.bpol_hat_kernel = pi_behavior

            sn_matrix = np.ones(shape=(self.N_hst, len(self.classes)))

            for c in self.classes:
                sn_matrix[:, c] = np.sum(self.A[:, c]/self.bpol_hat_kernel[:, c])

            return sn_matrix

        else:
            return self.N_hst

    def fit(self, folds=5, num_basis=False, sigma_list=None, lda_list=None, algorithm='Ridge', logit=False):
        x_train, x_test = self.X.T, self.Z.T
        t_train = self.Y
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

            cv_hst_fold = np.arange(folds) 

            cv_hst_split = np.floor(np.arange(self.N_hst)*folds/self.N_hst)
            cv_hst_index = cv_hst_split[np.random.permutation(self.N_hst)]
            
            cv_evl_split = np.floor(np.arange(self.N_evl)*folds/self.N_evl)
            cv_evl_index = cv_evl_split[np.random.permutation(self.N_evl)]

            x_cv = []
            a_cv = []
            y_cv = []
            z_cv = []
            p_evl_hst_cv = []
            p_evl_evl_cv = []
            
            for k in cv_hst_fold:
                x_cv.append(np.exp(-XC_dist[:, cv_index==k]/(2*sigma**2)))
                a_cv.append(self.A[cv_hst_index==k])
                y_cv.append(self.Y[cv_hst_index==k])
                z_cv.append(np.exp(-TC_dist[:, cv_index==k]/(2*sigma**2)))
                p_evl_hst_cv.append(self.pi_evaluation_train[cv_hst_index==k])
                p_evl_evl_cv.append(self.pi_evaluation_test[cv_evl_index==k])

            for k in range(folds):
                #print(h0_cv[0])
                # calculate the h vectors for training and test
                count = 0
                for j in range(folds):
                    if j == k:
                        x_te = x_cv[j]
                        y_te = y_cv[j]
                        z_te = z_cv[j]
い
                    if j != k:
                        if count == 0:
                            x_tr = x_cv[j]
                            a_tr = a_cv[j]
                            y_tr = y_cv[j]
                            z_tr = z_cv[j]
                            p_evl_hst_tr = p_evl_hst_cv[j]
                            p_evl_evl_tr = p_evl_evl_cv[j]
                            count += 1
                        else:
                            x_tr = np.append(x_tr, x_cv[j], axis=0)
                            a_tr = np.append(a_tr, a_cv[j], axis=0)
                            y_tr = np.append(y_tr, y_cv[j], axis=0)
                            z_tr = np.append(z_tr, z_cv[j], axis=0)
                            p_evl_hst_tr = np.append(p_evl_hst_tr, p_evl_hst_cv[j], axis=0)
                            p_evl_evl_tr = np.append(p_evl_evl_tr, p_evl_evl_cv[j], axis=0)

                    one_x = np.ones((len(x_tr),1))
                    one_z = np.ones((len(z_tr),1))
                    x_tr = np.concatenate([x_tr, one_x], axis=1)
                    z_tr = np.concatenate([z_tr, one_z], axis=1)
                    one_x = np.ones((len(x_te),1))
                    one_z = np.ones((len(z_te),1))
                    x_te = np.concatenate([x_te, one_x], axis=1)
                    z_te = np.concatenate([z_te, one_z], axis=1)

                    for lda_idx, lda in enumerate(lda_list):
                        beta = np.zeros(self.dim)


                        
                        clf.fit(htr, ttr.T)
                        pred = clf.predict(hte)
                        if logit:
                            score = -np.mean((pred == tte)**2)
                        else:
                            score = np.mean((pred - tte)**2)

            # pre-sum to speed up calculation
            x_cv = []
            a_cv = []
            y_cv = []
            z_cv = []
            p_evl_hst_cv = []
            p_evl_evl_cv = []
            
            for k in cv_hst_fold:
                x_cv.append(self.X[cv_hst_index==k])
                a_cv.append(self.A[cv_hst_index==k])
                y_cv.append(self.Y[cv_hst_index==k])
                z_cv.append(self.Z[cv_evl_index==k])
                p_evl_hst_cv.append(self.pi_evaluation_train[cv_hst_index==k])
                p_evl_evl_cv.append(self.pi_evaluation_test[cv_evl_index==k])

            for k in range(folds):
            #print(h0_cv[0])
                # calculate the h vectors for training and test
                count = 0
                for j in range(folds):
                    if j != k:
                        if count == 0:
                            x_tr = x_cv[j]
                            a_tr = a_cv[j]
                            y_tr = y_cv[j]
                            z_tr = z_cv[j]
                            p_evl_hst_tr = p_evl_hst_cv[j]
                            p_evl_evl_tr = p_evl_evl_cv[j]
                            count += 1
                        else:
                            x_tr = np.append(x_tr, x_cv[j], axis=0)
                            a_tr = np.append(a_tr, a_cv[j], axis=0)
                            y_tr = np.append(y_tr, y_cv[j], axis=0)
                            z_tr = np.append(z_tr, z_cv[j], axis=0)
                            p_evl_hst_tr = np.append(p_evl_hst_tr, p_evl_hst_cv[j], axis=0)
                            p_evl_evl_tr = np.append(p_evl_evl_tr, p_evl_evl_cv[j], axis=0)
                
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
                clf = linear_model.Lasso(lda_chosen)

        return clf, x_train, x_test

    def objective_function(self, x, z, beta):
        epol_hst = np.dot(x, beta) 
        epol_evl = np.dot(z, beta) 

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
