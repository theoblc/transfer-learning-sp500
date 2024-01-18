import numpy as np
import scipy as sp


def compute_V(X):
    return np.dot(np.transpose(X), X)

def compute_W(lamda, X_s, X_b, X_t):
     V_s = compute_V(X_s)
     V_b = compute_V(X_b)
     V_b_inv = np.linalg.inv(V_b)
     V_t = compute_V(X_t)

     W = np.dot(np.linalg.inv(V_s + lamda*(np.dot(np.dot(V_t,V_b_inv),V_s)) + lamda*V_t), V_s + lamda*(np.dot(np.dot(V_t,V_b_inv),V_s)))
     return W

def compute_M(X_s, X_b):
    V_s = compute_V(X_s)
    V_b = compute_V(X_b)
    V_b_inv = np.linalg.inv(V_b)
    V_s_sqrt = sp.linalg.sqrtm(V_s)
    
    M = np.dot(np.dot(V_s_sqrt, V_b_inv), V_s_sqrt)
    
    return M

def compute_U_eigenval(M):
    eigenvalues, eigenvectors = np.linalg.eig(M)

    U = eigenvectors
    diag_eigenval = np.diag(eigenvalues)
    nb_elt_on_diag = min(diag_eigenval.shape[0], diag_eigenval.shape[1])
    eigenvals = [diag_eigenval[i,i] for i in range(nb_elt_on_diag)]
    
    return U, eigenvals

def compute_beta_hat_s_or_b(X, Y):
    V = compute_V(X)
    V_inv = np.linalg.inv(V)
    
    beta_hat = np.dot(np.dot(V_inv, np.transpose(X)), Y)
    return beta_hat

# Formule du Lemme 2.1
def compute_beta_hat(W, beta_s, beta_b, d):
    beta_hat = np.dot(W, beta_s) + np.dot((np.eye(d) - W), beta_b)
    
    return beta_hat

# Formule du Lemme 2.1
def compute_gamma_hat(lamda, X_b, X_t, beta_hat_b, beta_hat):
    V_t = compute_V(X_t)
    V_b = compute_V(X_b)
    
    gamma_hat = np.dot(np.linalg.inv(V_b + lamda*V_t), np.dot(V_b, (beta_hat_b - beta_hat)))
    
    return gamma_hat

def compute_gamma_hat_bis(beta_hat_b, beta_hat_s):
    gamma_hat = beta_hat_b - beta_hat_s
    
    return gamma_hat

def compute_sigma2_hat(X, Y, beta_hat, n, d):
    # calcul de la norme au carré
    vector = Y - np.dot(X, beta_hat)
    sigma2_hat = np.dot(np.transpose(vector), vector)
    if (n > d):
        return sigma2_hat / (n-d)
    else:
        print("Error: n < d")
        return None
    
def compute_kappa2_hat(U, X_s, X_b, gamma_hat, sigma2_s_hat, sigma2_b_hat, d, method="plug"):
    """Argument method can be equal to : 'plug', 'bapi' or 'bapi_tild'."""
    V_s = compute_V(X_s)
    V_b = compute_V(X_b)
    V_s_inv = np.linalg.inv(V_s)
    V_b_inv = np.linalg.inv(V_b)
    V_s_sqrt = sp.linalg.sqrtm(V_s)
    
    if method == "plug":
        theta_hat = sigma2_b_hat*V_b_inv + np.dot(gamma_hat, np.transpose(gamma_hat))
    elif method == "bapi":
        zero_d = np.zeros((d, d))
        matrix = np.dot(gamma_hat, np.transpose(gamma_hat)) - sigma2_b_hat*V_b_inv - sigma2_s_hat*V_s_inv
        # Calcul de la partie positive (= partie positive de tous les éléments de la matrice)
        theta_hat = sigma2_b_hat*V_b_inv + np.maximum(matrix, zero_d)
    elif method == "bapi_tild":
        zero_d = np.zeros(d, d)
        matrix = np.dot(gamma_hat, np.transpose(gamma_hat)) - sigma2_s_hat*V_s_inv
        # Calcul de la partie positive (= partie positive de tous les éléments de la matrice)
        theta_hat = np.maximum(matrix, zero_d)
        
    kappa2_hat = np.zeros(d)
    for j in range(d):
        kappa2_hat[j] = np.transpose(U[:, j])@V_s_sqrt@theta_hat@V_s_sqrt@U[:,j]
    
    return kappa2_hat

def compute_EQM(lamda, sigma2_s_hat, sigma2_b_hat, eigenvals, X_s, X_b, X_t, beta_hat_s, beta_hat_b, U, d):
    W = compute_W(lamda, X_s, X_b, X_t)
    beta_hat = compute_beta_hat(W, beta_hat_s, beta_hat_b, d)
    gamma_hat = compute_gamma_hat(lamda, X_b, X_t, beta_hat_b, beta_hat)
    kappa2_hat = compute_kappa2_hat(U, X_s, X_b, gamma_hat, sigma2_s_hat, sigma2_b_hat, d, method="plug")
    
    vector = np.array([(sigma2_s_hat*(1+lamda*eigenvals[i])**2 + (lamda**2)*(kappa2_hat[i]**2)) / ((1+lamda+lamda*eigenvals[i])**2) for i in range(d)])
    eqm = np.sum(vector)
    
    return eqm

def compute_lamda(sigma2_s_hat, eigenvals, kappa2_hat, d):
    suum = np.sum([(sigma2_s_hat*eigenvals[i]*(3+4*eigenvals[i]) + kappa2_hat) for i in range(d)])
    
    lamda = d / suum
    return lamda