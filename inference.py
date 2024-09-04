import torch
from train import FeedForwardNet, downloadDatasets

classMapping = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]

def predict(model, input, target, classMapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predictedIndex = predictions[0].argmax(0)
        predicted = classMapping[predictedIndex]
        expected = classMapping[target]
    return predicted, expected
    
if __name__ == "__main__":
    
    # load model
    feedForwardNet = FeedForwardNet()
    stateDict = torch.load("feedforwardnet.pth")
    feedForwardNet.load_state_dict(stateDict)
    
    # load dataset
    _, validationData = downloadDatasets()
    
    # get sample from dataset for inference
    input, target = validationData[0][0], validationData[0][1]
    
    # make inference
    predicted, expected = predict(feedForwardNet, input, target, classMapping)
    
    print(f"Predicted: '{predicted}', expected: '{expected}'")