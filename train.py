import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.denseLayers = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, inputData):
        flatData = self.flatten(inputData)
        logits = self.denseLayers(flatData)
        predictions = self.softmax(logits)
        return predictions
    
def downloadDatasets():
    trainData = datasets.MNIST(
        root = "data",
        download = True,
        train = True,
        transform = ToTensor() 
    )
    validationData = datasets.MNIST(
        root = "data",
        download = True,
        train = False,
        transform = ToTensor() 
    )
    return trainData, validationData

def trainOneEpoch(model, dataLoader, lossFunc, optimizer, device):
    for inputs, targets in dataLoader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # calculate loss
        predictions = model(inputs)
        loss = lossFunc(predictions, targets)
        
        # backpropogate loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Loss: {loss.item()}")

def train(model, dataLoader, lossFunc, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        trainOneEpoch(model, dataLoader, lossFunc, optimizer, device)
        print("---------------")
    print("Training is done")
        

if __name__ == "__main__":
    trainData, _ = downloadDatasets()
    print("MNIST dataset downloaded")
    
    # create data loader
    trainDataLoader = DataLoader(trainData, batch_size=BATCH_SIZE)

    # build model
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")
    feedForwardNet = FeedForwardNet().to(device)
    
    lossFunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(feedForwardNet.parameters(), lr=LEARNING_RATE)

    # Train model
    train(feedForwardNet, trainDataLoader, lossFunc, optimizer, device, EPOCHS)
    
    torch.save(feedForwardNet.state_dict(), "feedforwardnet.pth")
    print("Model trained and stored at feedforwardnet.pth")