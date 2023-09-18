import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(784, 128, bias=False)  # compress to 128 features
        )
        self.decoder = nn.Sequential(             
            nn.Linear(128, 784, bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Data Preprocessing
trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./mnist_data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Lambda(lambda x: x.view(-1)), # flatten the image to a vector
                       transforms.Lambda(lambda x: x.float()),
            ])),
    batch_size=250, shuffle=True)

# read matrix from PCA
v_PCA = np.load('v_PCA.npy')

num_epochs = 40
model = Autoencoder()
distance = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

latent_space_errors = []

for epoch in range(num_epochs):
    for data in trainloader:
        img, _ = data
        # ===================forward=====================
        output = model(img.float())
        loss = distance(output, img.float())
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # get model weights for the first layer
        weights_encoder = model.encoder[0].weight.data.numpy().T

    # distance between encoder weights and v_PCA
    error = np.linalg.norm(weights_encoder - v_PCA)
    latent_space_errors.append(error)
    print('Latent Space-PCA Error = ', error)
    print('Epoch [{}/{}], Loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))

# plot latent space errors
plt.plot(latent_space_errors)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Latent Space-PCA Error')
plt.show()

testset = torchvision.datasets.MNIST('./mnist_data/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Lambda(lambda x: x.view(-1)), # flatten the image to a vector
                       transforms.Lambda(lambda x: x.float()),
            ]))

# test model on the first 5 images of test set
n = 5
testloader = torch.utils.data.DataLoader(testset, batch_size=n, shuffle=False)
dataiter = iter(testloader)
images, labels = dataiter.next()
outputs = model(images)

fig, axes = plt.subplots(nrows=2, ncols=n, sharex=True, sharey=True, figsize=(20,4))
for images, row in zip([images, outputs], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.view(28,28).detach().numpy(), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.show()

# calculate the mean square reconstruction error
testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False)
dataiter = iter(testloader)
images, labels = dataiter.next()
outputs = model(images)
error = torch.mean((images - outputs)**2)
print('Mean Reconstruction Error = ', error.item())
