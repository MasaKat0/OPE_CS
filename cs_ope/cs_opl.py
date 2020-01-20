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

    def dml_learning(self, self_norm=True, folds=2, method='Lasso'):
        theta_list = []
        
        cv_hst_fold = np.arange(folds) 
        cv_hst_split0 = np.floor(np.arange(self.N_hst)*folds/self.N_hst)
        cv_hst_index = cv_hst_split0[np.random.permutation(self.N_hst)]
        
        cv_evl_split0 = np.floor(np.arange(self.N_evl)*folds/self.N_evl)
        cv_evl_index = cv_evl_split0[np.random.permutation(self.N_evl)]

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
                        
            f_hst_matrix = np.zeros(shape=(len(x_tr), len(self.classes)))
            f_evl_matrix = np.zeros(shape=(len(z_tr), len(self.classes)))
            #w_matrix = np.zeros(shape=(len(x_tr), len(self.classes)))

            sn_matrix = np.ones(shape=(len(x_tr), len(self.classes)))

            _, a_temp = np.where(a_tr == 1)
            clf, x_ker_train, x_ker_test = KernelRegression(x_tr, a_temp, z_tr, algorithm=method, logit=True)
            clf.fit(x_ker_train, a_temp)
            p_bhv = clf.predict_proba(x_ker_train)

            w_matrix = p_evl_hst_tr/p_bhv
            
            for c in self.classes:
                clf, x_ker_train, x_ker_test = KernelRegression(x_tr, y_tr[:, c], z_tr, algorithm=method, logit=False)
                clf.fit(x_ker_train, y_tr[:, c])
                f_hst_matrix[:, c] = clf.predict(x_ker_train)
                f_evl_matrix[:, c] = clf.predict(x_ker_test)

                sn_matrix[:, c] = np.sum(a_tr[:, c]/p_bhv[:, c])
            
            densratio_obj = densratio(z_tr, x_tr)
            r = densratio_obj.compute_density_ratio(x_tr)
            r = np.array([r for c in self.classes]).T
            
            if self_norm:
                denominator = sn_matrix
            else:
                denominator = self.N_hst
            
            theta = np.sum(a_tr*(y_tr-f_hst_matrix)*w_matrix*r/denominator) + np.sum(f_evl_matrix*p_evl_evl_tr)/self.N_evl
            theta_list.append(theta)
        
        return np.mean(theta_list)

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
