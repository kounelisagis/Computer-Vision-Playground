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
            nn.Linear(784, 128)  # compress to 128 features
        )
        self.decoder = nn.Sequential(             
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward(self,x):
        # flatten image x to vector
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Data Preprocessing
trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./mnist_data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       transforms.Lambda(lambda x: x.view(-1)), # flatten the image to a vector
                       transforms.Lambda(lambda x: x/255), # normalize the image to [0, 1] - required by binary cross entropy
                       transforms.Lambda(lambda x: x.float()),
            ])),
    batch_size=250, shuffle=True)

# read matrix from PCA
v_PCA = np.load('v_PCA.npy')

num_epochs = 40
batch_size = 128
model = Autoencoder()
# distance = nn.MSELoss()
# distance = binary entropy loss
distance = nn.BCELoss()
# distance = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=1e-3)

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
