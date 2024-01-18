import numpy as np
import data_enriched_functions as enriched


class LinearRegressionTL:
    def __init__(self, d, method="plug"):
        self.d = d
        self.method = method
        
        self.n = 0
        self.N = 0
        
        self.M = 0
        self.U = 0
        self.eigenvals = 0
        
        self.lamda = 0
        self.beta_hat_s = 0
        self.beta_hat_b = 0
        self.beta_hat = 0
        
        self.sigma2_s_hat = 0
        self.sigma2_b_hat = 0
        
        self.gamma_hat = 0
        self.kappa2_hat = 0
        
    def fit(self, X_s, Y_s, X_b, Y_b):
        n = X_s.shape[0]
        self.n = n
        N = X_b.shape[0]
        self.N = N
        X_t = X_s

        M = enriched.compute_M(X_s, X_b)
        self.M = M

        U, eigenvals = enriched.compute_U_eigenval(M)
        self.U = U
        self.eigenvals = eigenvals

        beta_hat_s = enriched.compute_beta_hat_s_or_b(X_s, Y_s)
        self.beta_hat_s = beta_hat_s
        beta_hat_b = enriched.compute_beta_hat_s_or_b(X_b, Y_b)
        self.beta_hat_b = beta_hat_b
        
        sigma2_s_hat = enriched.compute_sigma2_hat(X_s, Y_s, self.beta_hat_s, n, self.d)
        self.sigma2_s_hat = sigma2_s_hat
        sigma2_b_hat = enriched.compute_sigma2_hat(X_b, Y_b, self.beta_hat_b, N, self.d)
        self.sigma2_b_hat = sigma2_b_hat
        
        gamma_hat = enriched.compute_gamma_hat_bis(self.beta_hat_b, self.beta_hat_s)
        self.gamma_hat = gamma_hat
        
        kappa2_hat = enriched.compute_kappa2_hat(U, X_s, X_b, self.gamma_hat, self.sigma2_s_hat, self.sigma2_b_hat, self.d, method=self.method)
        self.kappa2_hat = kappa2_hat
        
        lamda = enriched.compute_lamda(self.sigma2_s_hat, self.eigenvals, self.kappa2_hat, self.d)
        self.lamda = lamda
        
        W = enriched.compute_W(self.lamda, X_s, X_b, X_t)

        beta_hat = enriched.compute_beta_hat(W, self.beta_hat_s, self.beta_hat_b, self.d)
        self.beta_hat = beta_hat
        
        return self.beta_hat
        
        
    def predict(self, X_s_test):
        predicted = np.dot(X_s_test, self.beta_hat)
        
        return predicted