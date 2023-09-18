from torchvision import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA 
import seaborn as sns
import pandas as pd

train_dataset = datasets.MNIST(root='./mnist_data/', train=True, download=True)
sample_data = train_dataset.train_data[:5000]/255

# [60000, 28, 28] -> [60000, 784]
sample_data = sample_data.view(sample_data.size(0), -1).numpy()
print(sample_data.shape)

kpca = KernelPCA(kernel='rbf', n_components=700, fit_inverse_transform=True)
z = kpca.fit_transform(sample_data)
print(z.shape)

# reconstruct image 0
plt.imshow(kpca.inverse_transform([z[0]]).reshape(28,28), cmap='gray')
plt.show()

df = pd.DataFrame()
df["y"] = train_dataset.train_labels[:5000]
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]

sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 10),
                data=df).set(title="MNIST data KernelPCA projection")
plt.show()
