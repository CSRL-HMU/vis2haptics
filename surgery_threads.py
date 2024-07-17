import numpy as np
import cv2
import pyrealsense2 as rs
import threading
import time
import math
from collections import deque
from typing import Tuple
import torch
from PIL import Image
from data_loading import RegressionTaskData
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class CNNRegression(nn.Module):
    """
    This will be the very basic CNN model we will use for the regression task.
    """

    def __init__(self, image_size: Tuple[int, int, int] = (3, 620, 620), dropout_prob: float = 0.5):
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

        self.linear_line_size = int(64 * (image_size[1] // 8) * (image_size[2] // 8))
        self.fc1 = nn.Linear(in_features=self.linear_line_size, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.fc4 = nn.Linear(in_features=32, out_features=4)

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
            n_samples_total += len(output_angles)

        mean_loss = total_loss / len(regression_task.testloader)
        print(f'Test Loss: {mean_loss:.4f}')

        

def estimate_deform(img, model, device, image_size: Tuple[int, int, int] = (3, 620, 620)):
    """
    This estimates the deformation from an image file
    """
    if image_size[0] == 1:
        grayscale = True
    else:
        grayscale = False
    assert image_size[1] == image_size[2], 'Image size must be square'
    
    resize_size = image_size[1]
    regression_task = RegressionTaskData(grayscale=grayscale, resize_size=resize_size)

    test_tran = regression_task.test_transforms
    
    PIL_image = Image.fromarray(np.uint8(img)).convert('RGB')
    # print(PIL_image)
    in_image_tensor = test_tran(PIL_image)
    #
    im_transformed = in_image_tensor.numpy().transpose(1, 2, 0)
    # print(len(im_transformed[:][0][0]))
    # print('x')
    # print(len(im_transformed[0][:][0]))
    # print('x')
    # print(len(im_transformed[0][0][:]))

    # cv2.imshow("Realtime image", im_transformed)
    # cv2.waitKey(1)
    # plt.imshow(in_image_tensor.permute(1, 2, 0))


    # Evaluate the model on the test data
    with torch.no_grad():
   
        output = np.array(model(in_image_tensor.to(device)))
 
    
        # plt.plot([p0[0], p0[0]+output[0,0]], [p0[1], p0[1] + output[0,1]],'r')

    return output[0,:]

def calculate_norm(output, output_prev, d0):
    a_f = 0.99

    x1 = a_f * output[0] + (1.0 - a_f) * output_prev[0]
    x2 = a_f * output[1] + (1.0 - a_f) * output_prev[1]
    y1 = a_f * output[2] + (1.0 - a_f) * output_prev[2]
    y2 = a_f * output[3] + (1.0 - a_f) * output_prev[3]

    deform_filt = [x1, x2, y1, y2]
    distance = np.sqrt(((x2 - x1)-d0[0])**2 + ((y2 - y1)-d0[1])**2)

    return distance, deform_filt


class RealTimeProcessor:
    def __init__(self, model, device, image_size: Tuple[int, int, int] = (3, 620, 620)):
        self.model = model
        self.device = device
        self.image_size = image_size
        
        # Arxikopoihsh RealSense pipeline gia lhpsh eikonwn
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        
        self.time_list = deque(maxlen=100) # Keep the last 100 values
        self.norm_list = deque(maxlen=100)
        
        self.ptool = None
        self.d_thres = None
        
        self.is_running = False
        self.thread = threading.Thread(target=self._process_loop) # Arxikopoihsh thread
        self.lock = threading.Lock() 

    # Ekkinhsh tou thread    
    def start(self):
        self.is_running = True
        self.thread.start()

    # Stamathma tou thread   
    def stop(self):
        self.is_running = False
        self.thread.join()
        self.pipeline.stop()
        cv2.destroyAllWindows()

    # Epeksergasia thread
    def _process_loop(self):
        start_time = time.time()
        first_time_flag = False

        deform_filt = [0, 0, 0, 0]
        
        # plt.ion()
        # fig, ax = plt.subplots()
        # line, = ax.plot(self.time_list, self.norm_list)
        # ax.set_xlabel('Time (s)')
        # ax.set_ylabel('Norm')
        # ax.set_title('Real-time Norm vs Time')
        
        while self.is_running:

            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            img = np.asanyarray(color_frame.get_data())
            
            img_resized = img[math.floor(720/3):720, 426:1146]
            
            deform = estimate_deform(img, self.model, self.device)
            if not first_time_flag:
                first_time_flag = True
                d0 = [deform[1]-deform[0], deform[3]-deform[2]]

            # cv2.line(img_resized, (math.floor(deform[0]), math.floor(deform[2])), (math.floor(deform[1]), math.floor(deform[3])), (255, 0, 0), 2)
            # cv2.imshow("Realtime image", img_resized)
            # cv2.waitKey(1)
            
            norm, deform_filt = calculate_norm(deform, deform_filt, d0)
            
            ptool = np.array([deform_filt[1], deform_filt[3]])
            pnom = np.array([deform_filt[0], deform_filt[2]])
            
            d = np.array([deform_filt[1]-deform_filt[0], deform_filt[3]-deform_filt[2]])
            
            if norm < 5.0:
                d_thres = [0, 0]
                norm_thres = 0.0
            else:
                d_thres = d * (1.0 - 5.0/norm)
                norm_thres = norm - 5.0
                
            current_time = time.time() - start_time
            self.time_list.append(current_time)
            self.norm_list.append(norm_thres)
            
            # Update plot
            # line.set_xdata(self.time_list)
            # line.set_ydata(self.norm_list)
            # ax.relim()
            # ax.autoscale_view()
            # plt.draw()
            # plt.pause(0.01)
            
            # print(deform, norm_thres)

            with self.lock:
                self.ptool = ptool
                self.d_thres = d_thres
    
    def get_results(self):
        with self.lock:
            return self.ptool, self.d_thres


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    image_size: Tuple[int, int, int] = (3, 620, 620)
    filename = 'model_epoch_800.pth'
    
    # Load the model
    model = load_model(image_size=image_size, filename=filename)
    model.to(device)
    
    processor = RealTimeProcessor(model, device, image_size)
    processor.start()

    t_now = 0
    t_prev = 0
    
    try:
        while True:

            ptool, d_thres = processor.get_results()
            if ptool is not None and d_thres is not None:
                print(f'ptool: {ptool}, d_thres: {d_thres}')
                t_now = time.time()
                print(t_now - t_prev)
                t_prev = t_now
            # time.sleep(1)
    except KeyboardInterrupt:
        processor.stop()
