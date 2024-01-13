import numpy as np
import data_enriched as enriched


class LinearRegressionTL:
    def __init__(self, d, method):
        self.d = d
        self.method = method
        
        self.n = 0
        self.N = 0
        
        self.M = 0
        self.U = 0
        self.lamda = 0
        self.beta_hat = 0
        self.gamma_hat = 0
        self.kappa2_hat = 0
        
    def fit(self, X_s, Y_s, X_b, Y_b):
        n = X_s.shape[0]
        N = X_b.shape[0]
        X_t = X_s
        #print("n=",n,"N=", N)

        M = enriched.compute_M(X_s, X_b)
        self.M = M
        #print("M=", M)

        U, eigenvals = enriched.compute_U_eigenval(M)
        self.U = U
        #print("U=", U)
        #print("eigenvals=", eigenvals)

        beta_hat_s = enriched.compute_beta_hat_s_or_b(X_s, Y_s)
        #print("beta_hat_s=", beta_hat_s)
        beta_hat_b = enriched.compute_beta_hat_s_or_b(X_b, Y_b)
        #print("beta_hat_b=", beta_hat_b)

        sigma2_s_hat = enriched.compute_sigma2_hat(X_s, Y_s, beta_hat_s, n, self.d)
        #print("sigma2_s_hat=", sigma2_s_hat)
        sigma2_b_hat = enriched.compute_sigma2_hat(X_b, Y_b, beta_hat_b, N, self.d)
        #print("sigma2_b_hat=", sigma2_b_hat)
        
        lamda = 20 # /!\ Dans l'idéal il faudrait coder une descente de gradient pour trouver le lamda optimal /!\
        self.lamda = lamda
        
        W = enriched.compute_W(lamda, X_s, X_b, X_t)

        beta_hat = enriched.compute_beta_hat(W, beta_hat_s, beta_hat_b, self.d)
        self.beta_hat = beta_hat

        gamma_hat = enriched.compute_gamma_hat(lamda, X_b, X_t, beta_hat_b, beta_hat)
        self.gamma_hat = gamma_hat

        kappa2_hat = enriched.compute_kappa2_hat(U, X_s, X_b, gamma_hat, sigma2_s_hat, sigma2_b_hat, self.d, method=self.method)
        self.kappa2_hat = kappa2_hat
        
        
    def predict(self, X_s_test):
        predicted = np.dot(X_s_test, self.beta_hat)
        
        return predicted