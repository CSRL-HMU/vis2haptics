from typing import Tuple

import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import csv
 
import pyrealsense2 as rsu

from PIL import Image 

import time

from data_loading import RegressionTaskData


class CNNRegression(nn.Module):
    """
    This will be the very basic CNN model we will use for the regression task.
    """
    def __init__(self, image_size: Tuple[int, int, int] = (3, 620, 620), dropout_prob: float = 0.4):
        super(CNNRegression, self).__init__()
        self.image_size = image_size
        self.dropout_prob = dropout_prob

        self.conv1 = nn.Conv2d(in_channels=self.image_size[0], out_channels=4, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=self.dropout_prob)

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(p=self.dropout_prob)

        self.linear_line_size = int(64*(image_size[1]//8)*(image_size[2]//8))
        self.fc1 = nn.Linear(in_features=self.linear_line_size, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=16)
        self.fc4 = nn.Linear(in_features=16, out_features=4)

        
    def forward(self, x):
        """
        Passes the data through the network.
        There are commented out print statements that can be used to 
        check the size of the tensor at each layer. These are very useful when
        the image size changes and you want to check that the network layers are 
        still the correct shape.
        """
        # convolutional, activation, pooling, dropout, fully connected
        x = self.conv1(x)
        # print('Size of tensor after each layer')
        # print(f'conv1 {x.size()}')
        x = nn.functional.relu(x)
        # print(f'relu1 {x.size()}')
        x = self.pool1(x)
        # print(f'pool1 {x.size()}')
        x = self.dropout1(x)

        x = self.conv2(x)
        # print(f'conv2 {x.size()}')
        x = nn.functional.relu(x)
        # print(f'relu2 {x.size()}')
        x = self.pool2(x)
        # print(f'pool2 {x.size()}')
 
        # print(f'view1 {x.size()}')
        x = self.conv3(x)
        # print(f'conv2 {x.size()}')
        x = nn.functional.relu(x)
        # print(f'relu2 {x.size()}')
        x = self.pool3(x)
        # print(f'pool2 {x.size()}')
        x = self.dropout3(x)

        x = x.view(-1, self.linear_line_size)
        x = self.fc1(x)
        # print(f'fc1 {x.size()}')
        x = nn.functional.relu(x)
        # print(f'relu2 {x.size()}')
        x = self.fc2(x)
        # print(f'fc2 {x.size()}')
        x = self.fc3(x)
        # print(f'fc3 {x.size()}')
        x = nn.functional.relu(x)
        x = self.fc4(x)
        # print(f'fc4 {x.size()}')
        return x
    

def train_network(device, n_epochs: int = 10, image_size: Tuple[int, int, int] = (3, 620, 620)):
    """
    This trains the network for a set number of epochs.
    """
    if image_size[0] == 1:
        grayscale = True
    else:
        grayscale = False
    assert image_size[1] == image_size[2], 'Image size must be square'
    resize_size = image_size[1]
    regression_task = RegressionTaskData(grayscale=grayscale, resize_size=resize_size)

    # Define the model, loss function, and optimizer
    model = CNNRegression(image_size=image_size)
    model.to(device)
    print(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    flagBreak = False

    # Train the model
    writer = SummaryWriter()
    for epoch in tqdm(range(n_epochs)):

        t0 = time.time()

        for i, (inputs, targets) in enumerate(regression_task.trainloader):
            
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            writer.add_scalar('Train Loss', loss.item(), i)

            # Print training statistics
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{n_epochs}], Step [{i + 1}/{len(regression_task.trainloader)}], Loss: {loss.item():.4f}')

            
            # Save kathe 100 epoches
            if (epoch + 1) % 100 == 0:
                save_model(model, filename=f'model_epoch_{epoch + 1}.pth')
            
            # Kathe 100 epochs ananewsh
            if (epoch + 1) % 100 == 0:
                model = load_model(image_size=image_size, filename=f'model_epoch_{epoch + 1}.pth')

            # Stop when loss < 5
            if loss.item() < 5:
                save_model(model, filename=f'model_epoch_{epoch + 1}.pth')
                print(f'The loss < 5. The train stopped at epoch {epoch+1}')
                flagBreak = True
                break

        if flagBreak:
            break

        print('Time duration of this iteration:', time.time() - t0)
    writer.close()

    return model


def save_model(model, filename='3_620_620.pth'):
    """
    After training the model, save it so we can use it later.
    """
    torch.save(model.state_dict(), filename)


def load_model(image_size=(3, 620, 620), filename='3_620_620.pth'):
    """
    Load the model from the saved state dictionary.
    """
    model = CNNRegression(image_size)
    model.load_state_dict(torch.load(filename))
    return model


def evaluate_network(model, device, image_size: Tuple[int, int, int] = (3, 620, 620)):
    """
    This evaluates the network on the test data.
    """
    if image_size[0] == 1:
        grayscale = True
    else:
        grayscale = False
    assert image_size[1] == image_size[2], 'Image size must be square'
    resize_size = image_size[1]

    regression_task = RegressionTaskData(grayscale=grayscale, resize_size=resize_size)
    criterion = nn.MSELoss()

    # Evaluate the model on the test data
    with torch.no_grad():
        total_loss = 0
        total_angle_error = 0
        n_samples_total = 0
        for inputs, targets in regression_task.testloader:
            # Calculate the loss with the criterion we used in training
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))
            total_loss += loss.item()

            # We are actually predicting angles so we can calculate the angle error too
            # which is probably more meaningful to humans that the MSE loss
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            output_angles = np.array([np.arctan2(out[0], out[1]) for out in outputs_np])
            target_angles = np.array([np.arctan2(t[0], t[1]) for t in targets_np])
            # This is probably not a great way to calculate the angle error 
            # as it doesn't take into account the fact that angles wrap around
            # but it seems to work well enough for now
            angle_error = np.sum(np.abs(np.rad2deg(target_angles - output_angles)))
            total_angle_error += angle_error
            n_samples_total += len(output_angles)

        mean_loss = total_loss / len(regression_task.testloader)
        mean_angle_error = total_angle_error / n_samples_total
        print(f'Test Loss: {mean_loss:.4f}')
        print(f'Test mean angle error: {mean_angle_error:.4f} degrees')



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Train the model
    image_size: Tuple[int, int, int] = (3, 620, 620)
    model = train_network(device, 1200, image_size=image_size)###########################################################################################################

    # # Save the model
    filename = f'{image_size[0]}_{image_size[1]}_{image_size[2]}.pth'
    filename = '3_620_620.pth'
    save_model(model, filename=filename)

    # Load the model
    model = load_model(image_size=image_size, filename=filename)
    model.to(device)

    # Evaluate the model
    evaluate_network(model, device, image_size=image_size) 