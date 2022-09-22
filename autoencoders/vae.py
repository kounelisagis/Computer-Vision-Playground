import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encFC1 = nn.Linear(784, 256)
        self.encFC2 = nn.Linear(256, 64)
        self.encFC3mean = nn.Linear(64, 2)
        self.encFC3variance = nn.Linear(64, 2)

        self.decFC1 = nn.Linear(2, 64)
        self.decFC2 = nn.Linear(64, 256)
        self.decFC3 = nn.Linear(256, 784)

    def encoder(self, x):
        # the output feature map predicts mean (mu) and variance (logVar)
        # mu and logVar are used for generating middle representation and calculation of KL divergence loss
        x = F.relu(self.encFC1(x))
        x = F.relu(self.encFC2(x))
        mu = self.encFC3mean(x)
        logVar = self.encFC3variance(x)

        return mu, logVar

    def reparameterize(self, mu, logVar):
        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, x):
        # The generated output is the same size of the original input and in the range of [0, 1]
        x = F.relu(self.decFC1(x))
        x = F.relu(self.decFC2(x))
        x = torch.sigmoid(self.decFC3(x))
        return x

    def forward(self, x):
        # encoder -> reparameterization -> decoder
        mu, logVar = self.encoder(x)
        x = self.reparameterize(mu, logVar)
        x = self.decoder(x)
        return x, mu, logVar


# Data Preprocessing
trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./mnist_data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Lambda(lambda x: x.view(-1)), # flatten the image to a vector
                       transforms.Lambda(lambda x: x.float()),
            ])),
    batch_size=250, shuffle=True)

num_epochs = 100
model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print('Number of parameters of model: ', sum(p.numel() for p in model.parameters()))

for epoch in range(num_epochs):
    for data in trainloader:
        img, _ = data
        # ===================forward=====================
        output, mean, logVar = model(img.float())
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mean.pow(2) + logVar.exp())
        loss = F.binary_cross_entropy(output, img.float(), reduction='sum') + kl_divergence
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))

    # create a batch of noise~N(0,1)
    z = torch.randn(64, 2)
    output = model.decoder(z)

    # plot reconstructed images
    if epoch+1 in [1, 50, 100]:
        fig, axes = plt.subplots(nrows=8, ncols=8, sharex=True, sharey=True, figsize=(12,12))
        for ax, img in zip(axes.flatten(), output):
            ax.imshow(img.view(28,28).detach().numpy(), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()


# scatter plot of the latent space different color for different digits of the test set
testset = torchvision.datasets.MNIST('./mnist_data/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Lambda(lambda x: x.view(-1)), # flatten the image to a vector
                       transforms.Lambda(lambda x: x.float()),
            ]))

testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False)
dataiter = iter(testloader)
images, labels = dataiter.next()
outputs = model.encoder(images.float())
# keep only the mean
latent_vectors = outputs[0].detach().numpy()

plt.figure(figsize=(10, 10))
plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=labels, cmap='tab10')
plt.colorbar()
plt.show()
