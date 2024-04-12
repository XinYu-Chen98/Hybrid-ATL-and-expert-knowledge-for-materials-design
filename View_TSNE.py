import numpy as np
import matplotlib.pyplot as plt

def cal_pairwise_dist(x):
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    return dist
 
def cal_perplexity(dist, idx=0, beta=1.0):
    prob = np.exp(-dist * beta)
    prob[idx] = 0
    sum_prob = np.sum(prob)
    if sum_prob == 0:
        prob = np.maximum(prob, 1e-12)
        perp = -12
    else:
        prob /= sum_prob
        perp = 0
        for pj in prob:
            if pj != 0:
                perp += -pj*np.log(pj)
    return perp, prob
 
def seach_prob(x, tol=1e-5, perplexity=30.0):
    print("Computing pairwise distances...")
    (n, d) = x.shape
    dist = cal_pairwise_dist(x)
    pair_prob = np.zeros((n, n))
    beta = np.ones((n, 1))
    base_perp = np.log(perplexity)
 
    for i in range(n):
        if i % 500 == 0:
            print("Computing pair_prob for point %s of %s ..." %(i,n))
 
        betamin = -np.inf
        betamax = np.inf
        perp, this_prob = cal_perplexity(dist[i], i, beta[i])
 
        perp_diff = perp - base_perp
        tries = 0
        while np.abs(perp_diff) > tol and tries < 50:
            if perp_diff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2
 
            perp, this_prob = cal_perplexity(dist[i], i, beta[i])
            perp_diff = perp - base_perp
            tries = tries + 1
        pair_prob[i,] = this_prob
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    return pair_prob
 
def tsne(x, no_dims=2, initial_dims=15, perplexity=20.0, max_iter=300):
    """Runs t-SNE on the dataset in the NxD array x
    to reduce its dimensionality to no_dims dimensions.
    The syntaxis of the function is Y = tsne.tsne(x, no_dims, perplexity),
    where x is an NxD NumPy array.
    """
 
    if isinstance(no_dims, float):
        print("Error: array x should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1
 
    (n, d) = x.shape
    print (x.shape)
 
    eta = 500
    y = np.random.randn(n, no_dims)
    dy = np.zeros((n, no_dims))
    P = seach_prob(x, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)  
    P = P * 4
    P = np.maximum(P, 1e-12)
 
    for iter in range(max_iter):
        sum_y = np.sum(np.square(y), 1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(y, y.T), sum_y).T, sum_y))
        num[range(n), range(n)] = 0
        Q = num / np.sum(num)   #qij
        Q = np.maximum(Q, 1e-12)   
 
        # Compute gradient
        #pij-qij
        PQ = P - Q
        for i in range(n):
            dy[i,:] = np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (y[i,:] - y), 0)

        y = y - eta*dy
        y = y - np.tile(np.mean(y, 0), (n, 1))
        if (iter + 1) % 50 == 0:
            if iter > 100:
                C = np.sum(P * np.log(P / Q))
            else:
                C = np.sum( P/4 * np.log( P/4 / Q))
            print("Iteration ", (iter + 1), ": error is ", C)
        if iter == 100:
            P = P / 4
    print("finished training!")
    return y


h_s = np.load('sourceExtracted.npy').tolist()
h_t = np.load('targetExtracted.npy').tolist()


if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    X = []
    labels = []
    for i in range(1500):
        X.append(h_s[i])
        labels.append(1)
    for i in range(1500):
        X.append(h_t[i])
        labels.append(0)
    
    Y = tsne(np.array(X), 2, 15, 20.0)
    #print(Y[:, 1])



plt.figure(figsize=(7,6))
plt.scatter(Y[:,0], Y[:,1], 20, labels)
#np.save('t-SNE_ADV',Y)
plt.ylabel('t-SNE1')
plt.xlabel('t-SNE2')
plt.colorbar(label = 'Bulk/2D')
plt.show()