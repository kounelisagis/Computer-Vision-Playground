'''
Note: commented lines give more functionalities
'''

from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt

train_dataset = datasets.MNIST(root='./mnist_data/', train=True, download=True)

# keep only the images with label = 0 or 4 or 5
# sample_data = train_dataset.train_data[(train_dataset.train_labels == 0) | (train_dataset.train_labels == 4) | (train_dataset.train_labels == 5)]
sample_data = train_dataset.train_data

# print median of the dataset
# plt.imshow(np.average(sample_data, 0), cmap='gray')
# plt.show()

# [60000, 28, 28] -> [60000, 784]
sample_data = sample_data.view(sample_data.size(0), -1).numpy()

# calculate mean
mean = np.mean(sample_data, 0)
# subtract mean - VERY IMPORTANT - PCA works only on zero mean data
sample_data = sample_data - mean

# find the co-variance matrix which is A^T * A
covar_matrix = np.matmul(sample_data.T, sample_data)
# plot covar_matrix
# plt.imshow(covar_matrix, cmap='gray')
# plt.show()

# find the eigen values and eigen vectors
w, v = np.linalg.eig(covar_matrix)

# sort v in descending order acording to w
idx = w.argsort()[::-1]
v = v[:,idx].real

# plot first 8 eigen vectors same plot
# for i in range(8):
#     plt.subplot(2, 4, i+1)
#     plt.imshow(v[:,i].reshape(28, 28), cmap='gray')
# plt.show()

# principal_components = [1, 8, 16, 64, 256]
principal_components = [128]
plt.subplot(1, len(principal_components)+1, 1)
plt.title('Original')
plt.imshow((sample_data[0]+mean).reshape(28, 28), cmap='gray')

errors = []

for i, L in enumerate(principal_components):
    # get the top L eigen vectors
    v_hat = v[:, :L]
    # save them to file
    np.save('v_PCA.npy', v_hat)

    # reconstruct the data using top L eigen vectors
    reconstructed_data = np.matmul(sample_data, v_hat)
    reconstructed_data = np.matmul(reconstructed_data, v_hat.T)
    # complex to float64
    reconstructed_data = np.real(reconstructed_data)

    # plot sample_data[0] and new_data[0] in the same plot
    plt.subplot(1, len(principal_components)+1, i+2)
    plt.title('L = {}'.format(L))
    plt.imshow((reconstructed_data[0]+mean).reshape(28, 28), cmap='gray')

    # calculate the error between sample_data and reconstructed_data
    error = np.average(np.square(sample_data - reconstructed_data))
    errors.append((L, error))
plt.show()

indices = np.arange(len(errors))
plt.bar(indices, [error[1] for error in errors])
plt.xticks(indices, [error[0] for error in errors])
plt.xlabel('L')
plt.ylabel('Mean Square error')
plt.show()
