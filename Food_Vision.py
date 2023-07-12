import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 1. Getting the Data Ready
# Define the transformations: Resize the image to 224x224 pixels (since this is a common size for image recognition models)
# and convert it to a PyTorch tensor
transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()])

# Load the Food101 dataset using torchvision's ImageFolder utility.
# This assumes the dataset is organized in a directory structure where each subdirectory represents a class, and contains
# images of that class. Replace 'path_to_food101' with the actual path to the Food101 dataset on your local machine.
train_data = datasets.ImageFolder('path_to_food101/train', transform=transform)
test_data = datasets.ImageFolder('path_to_food101/test', transform=transform)

# Create data loaders to allow batch processing of these images
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

# 2. Building the Model
# Define a custom neural network for classifying the food images. The network uses three convolutional layers for feature extraction,
# followed by two fully connected layers for classification.
class FoodNet(nn.Module):
    def __init__(self):
        super(FoodNet, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        # Define the fully connected layers
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 101) # 101 classes in Food101

    def forward(self, x):
        # Pass the input through the convolutional layers with ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        # Flatten the output of the convolutional layers
        x = x.view(-1, 64 * 28 * 28)
        # Pass the flattened output through the fully connected layers with ReLU activation on the first
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
model = FoodNet()

# 3. Fitting the Model to the Data
# Define the loss function (CrossEntropyLoss is commonly used for classification problems)
criterion = nn.CrossEntropyLoss()
# Define the optimizer (Stochastic Gradient Descent in this case)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Define the training loop
def train(model, train_loader, criterion, optimizer):
    # Loop over the dataset multiple times
    for epoch in range(10):
        running_loss = 0.0
        # Iterate over the data in the DataLoader
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

# 4. Making Predictions and Evaluating the Model
# Function to evaluate the model on the test set
def test(model, test_loader):
    correct = 0
    total = 0
    # No need to track gradients for testing, so wrap in no_grad()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on test images: %d %%' % (100 * correct / total))

# Run the training and testing
train(model, train_loader, criterion, optimizer)
test(model, test_loader)

# 5. Saving and Loading a Model
# Save the model parameters to a file
torch.save(model.state_dict(), 'model.pth')