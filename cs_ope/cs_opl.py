import numpy as np
import pandas as pdå
import statsmodels.api as sm

from kernel_regression import KernelRegression
from statsmodels.nonparametric.kernel_regression import KernelReg
from sklearn.neighbors import KernelDensity
from densratio import densratio
from scipy.optimize import minimize

from sklearn.kernel_ridge import KernelRidge


class op_learning():
    def __init__(self, X, A, Y_matrix, Z, classes):
        self.X = X
        self.A = A
        self.Y = Y_matrix
        self.Z = Z
        self.classes = classes

        self.N_hst, self.dim = X.shape
        self.N_evl = len(Z)

        self.f_hat_kernel = None
        self.bpol_hat_kernel = None
        self.q_hat_kernel = None
        self.p_hat_kernel = None
        self.prepare = False

    def ipw_est_parameters(self):
        if self.p_hat_kernel is None:
            dens = sm.nonparametric.KDEMultivariate(data=self.X, var_type='c'*self.dim, bw='normal_reference')
            self.p_hat_kernel = dens.pdf(self.X)
            
        if self.q_hat_kernel is None:
            dens = sm.nonparametric.KDEMultivariate(data=self.Z, var_type='c'*self.dim, bw='normal_reference')
            self.q_hat_kernel = dens.pdf(self.X)
            
        if self.bpol_hat_kernel is None:
            pi_behavior = np.zeros(shape=(self.N_hst, len(self.classes)))
            for c in self.classes:
                while True:
                    perm = np.random.permutation(self.N_hst)
                    A_temp = self.A[perm[:100]]
                    X_temp = self.X[perm[:100]]
                    model = KernelReg(A_temp[:,c], X_temp, var_type='c'*self.dim, reg_type='lc')
                    #model = KernelReg(self.A[:,c], self.X, var_type='c'*self.dim, reg_type='lc', bw=model.bw)
                    #model = KernelReg(self.A[:,c], self.X, var_type='c'*self.dim, reg_type='lc')
                    mu, _ = model.fit(self.X)
                    mu[mu < 0.001] = 0.001
                    mu[mu > 0.999] = 0.999
                    pi_behavior[:, c] = mu
                    if len(mu[~((mu > -100)&(mu < 100))]) == 0:
                        break

            self.bpol_hat_kernel = pi_behavior
        
        r = self.q_hat_kernel/self.p_hat_kernel
        r = np.array([r for c in range(len(self.classes))]).T

        sn_matrix = np.ones(shape=(self.N_hst, len(self.classes)))

        for c in self.classes:
            sn_matrix[:, c] = np.sum(self.A[:, c]/self.bpol_hat_kernel[:, c])
        
        self.r_ker_matrix = r
        self.sn_ker_matrix = sn_matrix

    def ipw_fit(self, folds=5, num_basis=False, sigma_list=None, lda_list=None, algorithm='Ridge', self_norm=False):
        x_train, x_test = self.X.T, self.Z.T
        XC_dist, TC_dist, CC_dist, n, num_basis = dist(x_train, x_test, num_basis)
        # setup the cross validation
        cv_hst_fold = np.arange(folds) 

        cv_hst_split = np.floor(np.arange(self.N_hst)*folds/self.N_hst)
        cv_hst_index = cv_hst_split[np.random.permutation(self.N_hst)]
        
        # set the sigma list and lambda list
        if sigma_list==None:
            sigma_list = np.array([0.001, 0.01, 0.1, 1, 10])
        if lda_list==None:
            lda_list = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1.,])

        score_cv = np.zeros((len(sigma_list), len(lda_list)))

        for sigma_idx, sigma in enumerate(sigma_list):
            x_cv = []
            a_cv = []
            y_cv = []
            p_bhv_hst_cv = []
            r_hst_cv = []
            
            for k in cv_hst_fold:
                x_cv.append(np.exp(-XC_dist[:, cv_hst_index==k]/(2*sigma**2)))
                a_cv.append(self.A[cv_hst_index==k])
                y_cv.append(self.Y[cv_hst_index==k])
                p_bhv_hst_cv.append(self.bpol_hat_kernel[cv_hst_index==k])
                r_hst_cv.append(self.r_ker_matrix[cv_hst_index==k])

            for k in range(folds):
                #print(h0_cv[0])
                # calculate the h vectors for training and test
                count = 0
                for j in range(folds):
                    if j == k:
                        x_te = x_cv[j].T
                        a_te = a_cv[j]
                        y_te = y_cv[j]
                        p_bhv_hst_te = p_bhv_hst_cv[j]
                        r_hst_te = r_hst_cv[j]
                    
                    if j != k:
                        if count == 0:
                            x_tr = x_cv[j].T
                            a_tr = a_cv[j]
                            y_tr = y_cv[j]
                            p_bhv_hst_tr = p_bhv_hst_cv[j]
                            r_hst_tr = r_hst_cv[j]
                            count += 1
                        else:
                            x_tr = np.append(x_tr, x_cv[j].T, axis=0)
                            a_tr = np.append(a_tr, a_cv[j], axis=0)
                            y_tr = np.append(y_tr, y_cv[j], axis=0)
                            p_bhv_hst_tr = np.append(p_bhv_hst_tr, p_bhv_hst_cv[j], axis=0)
                            r_hst_tr = np.append(r_hst_tr, r_hst_cv[j], axis=0)

                one_x = np.ones((len(x_tr),1))
                x_tr = np.concatenate([x_tr, one_x], axis=1)
                one_x = np.ones((len(x_te),1))
                x_te = np.concatenate([x_te, one_x], axis=1)

                for lda_idx, lda in enumerate(lda_list):
                    beta = np.zeros(shape=(x_tr.shape[1], len(self.classes)))
                    print(r_hst_tr.shape)
                    print(beta.shape)
                    f = lambda b: self.ipw_estimator(x_tr, a_tr, y_tr, p_bhv_hst_tr, r_hst_tr, b, lmd=lda, self_norm=self_norm)
                    res = minimize(f, beta)
                    beta = res.x
                    score0 = - self.ipw_estimator(x_tr, a_tr, y_tr, p_bhv_hst_tr, r_hst_tr, beta, lmd=0, self_norm=self_norm)
                    score = - self.ipw_estimator(x_te, a_te, y_te, p_bhv_hst_te, r_hst_te, beta, lmd=0, self_norm=self_norm)

                    print('score0', score0)
                    print('score', score)
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

        beta = np.zeros(shape=(x_train.shape[1], len(self.classes)))
        f = lambda b: self.ipw_estimator(x_train, self.A, self.Y, self.bpol_hat_kernel, self.r_ker_matrix, b, lmd=lda_chosen, self_norm=self_norm)
        res = minimize(f, beta)
        beta = res.x
        beta_list = beta.reshape(x_train.shape[1], len(self.classes))

        epol_evl = np.exp(-np.dot(x_test, beta_list))
        epol_evl = (epol_evl.T/np.sum(epol_evl, axis=1)).T

        return epol_evl

    def dm_est_parameters(self):
        f_matrix = np.zeros(shape=(self.N_evl, len(self.classes)))

        if self.f_hat_kernel is None:
            for c in self.classes:
                while True:
                    perm = np.random.permutation(self.N_hst)
                    Y_temp = self.Y[perm[:100]]
                    X_temp = self.X[perm[:100]]
                    model = KernelReg(Y_temp[:,c], X_temp, var_type='c'*self.dim, reg_type='lc')
                    #model = KernelReg(self.Y[:,c], self.X, var_type='c'*self.dim, reg_type='lc', bw=model.bw)
                    mu, _ = model.fit(self.Z)
                    mu[mu < 0.001] = 0.001
                    mu[mu > 0.999] = 0.999
                    f_matrix[:, c] = mu
                    if len(mu[~((mu > -100)&(mu < 100))]) == 0:
                        break
            
            self.f_hat_kernel = f_matrix

    def dm_fit(self, folds=5, num_basis=False, sigma_list=None, lda_list=None, algorithm='Ridge'):
        x_train, x_test = self.X.T, self.Z.T

        XC_dist, TC_dist, CC_dist, n, num_basis = dist(x_train, x_test, num_basis)
        # setup the cross validation
        cv_hst_fold = np.arange(folds) 
        
        cv_evl_split = np.floor(np.arange(self.N_evl)*folds/self.N_evl)
        cv_evl_index = cv_evl_split[np.random.permutation(self.N_evl)]

        # set the sigma list and lambda list
        if sigma_list==None:
            sigma_list = np.array([0.001, 0.01, 0.1, 1, 10])
        if lda_list==None:
            lda_list = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1.])

        score_cv = np.zeros((len(sigma_list), len(lda_list)))

        for sigma_idx, sigma in enumerate(sigma_list):

            z_cv = []
            f_evl_cv = []
            
            for k in cv_hst_fold:
                z_cv.append(np.exp(-TC_dist[:, cv_evl_index==k]/(2*sigma**2)))
                f_evl_cv.append(self.f_hat_kernel[cv_evl_index==k])

            for k in range(folds):
                #print(h0_cv[0])
                # calculate the h vectors for training and test
                count = 0
                for j in range(folds):
                    if j == k:
                        z_te = z_cv[j].T
                        f_evl_te = f_evl_cv[j]
                    
                    if j != k:
                        if count == 0:
                            z_tr = z_cv[j].T
                            f_evl_tr = f_evl_cv[j]
                            count += 1
                        else:
                            z_tr = np.append(z_tr, z_cv[j].T, axis=0)
                            f_evl_tr = np.append(f_evl_tr, f_evl_cv[j], axis=0)

                one_z = np.ones((len(z_tr),1))
                z_tr = np.concatenate([z_tr, one_z], axis=1)
                one_z = np.ones((len(z_te),1))
                z_te = np.concatenate([z_te, one_z], axis=1)

                for lda_idx, lda in enumerate(lda_list):
                    beta = np.zeros(shape=(z_tr.shape[1], len(self.classes)))
                    f = lambda b: self.reg_estimator(z_tr, f_evl_tr, b, lmd=lda)
                    res = minimize(f, beta)
                    beta = res.x
                    score0 = - self.reg_estimator(z_tr, f_evl_tr, beta, lmd=0.)
                    score = - self.reg_estimator(z_te, f_evl_te, beta, lmd=0.)

                    print('score0', score0)
                    print('score', score)
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
            
        beta = np.zeros(shape=(x_train.shape[1], len(self.classes)))
        f = lambda b: self.reg_estimator(x_test, self.f_hat_kernel, b, lmd=lda_chosen)
        res = minimize(f, beta)
        beta = res.x
        beta_list = beta.reshape(x_train.shape[1], len(self.classes))

        epol_evl = np.exp(-np.dot(x_test, beta_list))
        epol_evl = (epol_evl.T/np.sum(epol_evl, axis=1)).T

        return epol_evl

    def dml_est_parameters(self, folds=2, method='Ridge'):
        self.r_array = np.zeros((self.N_hst, len(self.classes)))
        self.f_hst_array = np.zeros((self.N_hst, len(self.classes)))
        self.f_evl_array = np.zeros((self.N_evl, len(self.classes)))
        self.bpol_array = np.zeros((self.N_hst, len(self.classes)))
        self.sn_hst_array = np.zeros((self.N_hst, len(self.classes)))
        
        cv_hst_fold = np.arange(folds) 

        cv_hst_split = np.floor(np.arange(self.N_hst)*folds/self.N_hst)
        perm_hst = np.random.permutation(self.N_hst)
        cv_hst_index = cv_hst_split[perm_hst]
        perm_hst = np.array([i for i in range(len(cv_hst_index))])
        
        cv_evl_split = np.floor(np.arange(self.N_evl)*folds/self.N_evl)
        perm_evl = np.random.permutation(self.N_evl)
        cv_evl_index = cv_evl_split[perm_evl]
        perm_evl = np.array([i for i in range(len(cv_evl_index))])

        x_cv = []
        a_cv = []
        y_cv = []
        z_cv = []
        perm_hst_cv = []
        perm_evl_cv = []
        
        for k in cv_hst_fold:
            x_cv.append(self.X[cv_hst_index==k])
            a_cv.append(self.A[cv_hst_index==k])
            y_cv.append(self.Y[cv_hst_index==k])
            z_cv.append(self.Z[cv_evl_index==k])
            perm_hst_cv.append(perm_hst[cv_hst_index==k])
            perm_evl_cv.append(perm_evl[cv_evl_index==k])

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
                        perm_hst_tr = perm_hst_cv[j]
                        perm_evl_tr = perm_evl_cv[j]
                        count += 1
                    else:
                        x_tr = np.append(x_tr, x_cv[j], axis=0)
                        a_tr = np.append(a_tr, a_cv[j], axis=0)
                        y_tr = np.append(y_tr, y_cv[j], axis=0)
                        z_tr = np.append(z_tr, z_cv[j], axis=0)
                        perm_hst_tr = np.append(perm_hst_tr, perm_hst_cv[j], axis=0)
                        perm_evl_tr = np.append(perm_evl_tr, perm_evl_cv[j], axis=0)
                        
            f_hst_matrix = np.zeros(shape=(len(x_tr), len(self.classes)))
            f_evl_matrix = np.zeros(shape=(len(z_tr), len(self.classes)))

            _, a_temp = np.where(a_tr == 1)
            clf, x_ker_train, x_ker_test = KernelRegression(x_tr, a_temp, z_tr, algorithm=method, logit=True)
            clf.fit(x_ker_train, a_temp)
            p_bhv = clf.predict_proba(x_ker_train)
            
            for c in self.classes:
                clf, x_ker_train, x_ker_test = KernelRegression(x_tr, y_tr[:, c], z_tr, algorithm=method, logit=False)
                clf.fit(x_ker_train, y_tr[:, c])
                f_hst_matrix[:, c] = clf.predict(x_ker_train)
                f_evl_matrix[:, c] = clf.predict(x_ker_test)
            
            densratio_obj = densratio(z_tr, x_tr)
            r = densratio_obj.compute_density_ratio(x_tr)
            r = np.array([r for c in self.classes]).T

            self.r_array[perm_hst_tr] = r
            self.f_hst_array[perm_hst_tr] = f_hst_matrix
            self.f_evl_array[perm_evl_tr] = f_evl_matrix
            self.bpol_array[perm_hst_tr] = p_bhv
        
        self.prepare = True

    def dml_fit(self, folds=5, num_basis=False, sigma_list=None, lda_list=None, algorithm='Ridge', self_norm=False):
        x_train, x_test = self.X.T, self.Z.T
        XC_dist, TC_dist, CC_dist, n, num_basis = dist(x_train, x_test, num_basis)
        # setup the cross validation
        cv_hst_fold = np.arange(folds) 

        cv_hst_split = np.floor(np.arange(self.N_hst)*folds/self.N_hst)
        cv_hst_index = cv_hst_split[np.random.permutation(self.N_hst)]
        
        cv_evl_split = np.floor(np.arange(self.N_evl)*folds/self.N_evl)
        cv_evl_index = cv_evl_split[np.random.permutation(self.N_evl)]

        # set the sigma list and lambda list
        if sigma_list==None:
            sigma_list = np.array([0.001, 0.01, 0.1, 1, 10])
        if lda_list==None:
            lda_list = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1.])

        score_cv = np.zeros((len(sigma_list), len(lda_list)))

        for sigma_idx, sigma in enumerate(sigma_list):

            x_cv = []
            a_cv = []
            y_cv = []
            z_cv = []
            f_hst_cv = []
            f_evl_cv = []
            p_bhv_hst_cv = []
            r_hst_cv = []
            
            for k in cv_hst_fold:
                x_cv.append(np.exp(-XC_dist[:, cv_hst_index==k]/(2*sigma**2)))
                print((np.exp(-XC_dist[:, cv_hst_index==k]/(2*sigma**2))).shape)
                a_cv.append(self.A[cv_hst_index==k])
                y_cv.append(self.Y[cv_hst_index==k])
                z_cv.append(np.exp(-TC_dist[:, cv_evl_index==k]/(2*sigma**2)))
                f_hst_cv.append(self.f_hst_array[cv_hst_index==k])
                f_evl_cv.append(self.f_evl_array[cv_evl_index==k])
                p_bhv_hst_cv.append(self.bpol_array[cv_hst_index==k])
                r_hst_cv.append(self.r_array[cv_hst_index==k])

            for k in range(folds):
                #print(h0_cv[0])
                # calculate the h vectors for training and test
                count = 0
                for j in range(folds):
                    if j == k:
                        x_te = x_cv[j].T
                        a_te = a_cv[j]
                        y_te = y_cv[j]
                        z_te = z_cv[j].T
                        f_hst_te = f_hst_cv[j]
                        f_evl_te = f_evl_cv[j]
                        p_bhv_hst_te = p_bhv_hst_cv[j]
                        r_hst_te = r_hst_cv[j]
                    
                    if j != k:
                        if count == 0:
                            x_tr = x_cv[j].T
                            a_tr = a_cv[j]
                            y_tr = y_cv[j]
                            z_tr = z_cv[j].T
                            f_hst_tr = f_hst_cv[j]
                            f_evl_tr = f_evl_cv[j]
                            p_bhv_hst_tr = p_bhv_hst_cv[j]
                            r_hst_tr = r_hst_cv[j]
                            count += 1
                        else:
                            x_tr = np.append(x_tr, x_cv[j].T, axis=0)
                            a_tr = np.append(a_tr, a_cv[j], axis=0)
                            y_tr = np.append(y_tr, y_cv[j], axis=0)
                            z_tr = np.append(z_tr, z_cv[j].T, axis=0)
                            f_hst_tr = np.append(f_hst_tr, f_hst_cv[j], axis=0)
                            f_evl_tr = np.append(f_evl_tr, f_evl_cv[j], axis=0)
                            p_bhv_hst_tr = np.append(p_bhv_hst_tr, p_bhv_hst_cv[j], axis=0)
                            r_hst_tr = np.append(r_hst_tr, r_hst_cv[j], axis=0)

                print(folds)
                print(j)
                print(count)
                one_x = np.ones((len(x_tr),1))
                one_z = np.ones((len(z_tr),1))
                print(x_tr.shape)
                x_tr = np.concatenate([x_tr, one_x], axis=1)
                print(x_tr.shape)
                z_tr = np.concatenate([z_tr, one_z], axis=1)
                one_x = np.ones((len(x_te),1))
                one_z = np.ones((len(z_te),1))
                x_te = np.concatenate([x_te, one_x], axis=1)
                z_te = np.concatenate([z_te, one_z], axis=1)

                for lda_idx, lda in enumerate(lda_list):
                    beta = np.zeros(shape=(x_tr.shape[1], len(self.classes)))
                    print(r_hst_tr.shape)
                    print(beta.shape)
                    f = lambda b: self.dm_estimator(x_tr, a_tr, y_tr, z_tr, f_hst_tr, f_evl_tr, p_bhv_hst_tr, r_hst_tr, b, lmd=lda, self_norm=self_norm)
                    res = minimize(f, beta)
                    beta = res.x
                    score0 = - self.dm_estimator(x_tr, a_tr, y_tr, z_tr, f_hst_tr, f_evl_tr, p_bhv_hst_tr, r_hst_tr, beta, lmd=0, self_norm=self_norm)
                    score = - self.dm_estimator(x_te, a_te, y_te, z_te, f_hst_te, f_evl_te, p_bhv_hst_te, r_hst_te, beta, lmd=0, self_norm=self_norm)

                    print('score0', score0)
                    print('score', score)
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

        beta = np.zeros(shape=(x_train.shape[1], len(self.classes)))
        f = lambda b: self.dm_estimator(x_train, self.A, self.Y, x_test, self.f_hst_array, self.f_evl_array, self.bpol_array, self.r_array, b, lmd=lda_chosen, self_norm=self_norm)
        res = minimize(f, beta)
        beta = res.x
        beta_list = beta.reshape(x_train.shape[1], len(self.classes))

        epol_evl = np.exp(-np.dot(x_test, beta_list))
        epol_evl = (epol_evl.T/np.sum(epol_evl, axis=1)).T

        return epol_evl

    def dm_objective_function(self, x, a, y, z, f_hst, f_evl, bpol, r, beta, self_norm=False):
        epol_hst = np.exp(-np.dot(x, beta))
        epol_hst = (epol_hst.T/np.sum(epol_hst, axis=1)).T
        epol_evl = np.exp(-np.dot(z, beta))
        epol_evl = (epol_evl.T/np.sum(epol_evl, axis=1)).T
        w = epol_hst/bpol

        sn_matrix = np.ones(shape=(len(x), len(self.classes)))
        if self_norm is True:
            for c in self.classes:
                sn_matrix[:, c] = np.sum(a[:, c]/bpol[:, c])
        else:
            sn_matrix /= len(x)

        theta = np.sum(a*(y-f_hst)*w*r/len(x)) + np.sum(f_evl*epol_evl)/len(f_evl)

        return theta

    def dm_estimator(self, x, a, y, z, f_hst, f_evl, bpol, r, beta, lmd=0., self_norm=False):
        beta_list = beta.reshape(x.shape[1], len(self.classes))

        return -self.dm_objective_function(x, a, y, z, f_hst, f_evl, bpol, r, beta_list, self_norm=self_norm) + lmd*np.sum(beta**2)

    def reg_estimator(self, z, f, beta, lmd=0.):
        beta_list = beta.reshape(z.shape[1], len(self.classes))

        return - self.reg_objective_function(z, f, beta_list) + lmd*np.sum(beta**2)

    def reg_objective_function(self, z, f, beta):
        epol_evl = np.exp(-np.dot(z, beta))
        epol_evl = (epol_evl.T/np.sum(epol_evl, axis=1)).T

        return np.sum(f*epol_evl)/len(z)

    def ipw_objective_function(self, x, a, y, bpol, r, beta, self_norm=False):
        epol_hst = np.exp(-np.dot(x, beta))
        epol_hst = (epol_hst.T/np.sum(epol_hst, axis=1)).T
        
        w = epol_hst/bpol

        sn_matrix = np.ones(shape=(len(x), len(self.classes)))
        if self_norm is True:
            for c in self.classes:
                sn_matrix[:, c] = np.sum(a[:, c]/bpol[:, c])
        else:
            sn_matrix /= len(x)

        w = epol_hst/bpol

        theta = np.sum(a*y*w*r/sn_matrix)
        
        return theta

    def ipw_estimator(self, x, a, y, bpol, r, beta, lmd=0., self_norm=False):
        beta_list = beta.reshape(x.shape[1], len(self.classes))

        return -self.ipw_objective_function(x, a, y, bpol, r, beta_list, self_norm=self_norm) + lmd*np.sum(beta**2)

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
