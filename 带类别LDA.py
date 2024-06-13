import numpy as np
import fc
def lda(X, y, n_components):
    """
    X: shape (n_samples, n_features)
    y: shape (n_samples, )
    n_components: int, the number of components to keep
    """
    # Calculate class means
    class_means = np.array([np.mean(X[y == i], axis=0) for i in np.unique(y)])
    
    # Calculate within-class scatter matrix
    Sw = np.zeros((X.shape[1], X.shape[1]))
    for i in np.unique(y):
        Xi = X[y == i]
        Sw += np.dot((Xi - class_means[i]).T, (Xi - class_means[i]))
    
    # Calculate between-class scatter matrix
    Sb = np.zeros((X.shape[1], X.shape[1]))
    for i in np.unique(y):
        Ni = np.sum(y == i)
        Sb += Ni * np.outer(class_means[i] - np.mean(X, axis=0), class_means[i] - np.mean(X, axis=0))
    
    # Solve eigenvalue problem for Sw^-1 * Sb
    eigvals, eigvecs = np.linalg.eig(np.dot(np.linalg.inv(Sw), Sb))
    
    # Sort eigenvectors by eigenvalues in decreasing order
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    
    # Project data onto the top n_components eigenvectors
    W = eigvecs[:, :n_components]
    X_lda = np.dot(X, W)
    
    return X_lda
# Generate toy data
X1 = np.random.randn(50, 2) + np.array([0, 5])
X2 = np.random.randn(50, 2) + np.array([5, 0])
X3 = np.random.randn(50, 2) + np.array([5, 5])
X = np.vstack([X1, X2, X3])
y = np.repeat([0, 1, 2], 50)

# Apply LDA
X_lda = lda(X, y, 1)

# Plot results
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
scatter = ax.scatter(X_lda, np.zeros_like(X_lda), c=y, cmap='viridis')
legend = ax.legend(*scatter.legend_elements(), title='Classes')
ax.add_artist(legend)
plt.show()
