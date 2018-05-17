# References
# https://github.com/hminle/car-behavioral-cloning-with-pytorch
import os
import fire
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torchvision.transforms as transforms
from drive_dataset import DriveData_LMDB
from drive_dataset import AugmentDrivingTransform
from drive_dataset import DrivingDataToTensor
from torch.utils.data import DataLoader
from model import CNNDriver

# Library that gives support for tensorboard and pytorch
from tensorboardX import SummaryWriter

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)


class Train:
    __device = []
    __writer = []
    __model = []
    __transformations = []
    __dataset_train = []
    __train_loader = []
    __loss_func = []
    __optimizer = []
    __exp_lr_scheduler = []

    def __init__(self):
        # Device configuration
        self.__device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.__writer = SummaryWriter('logs')
        self.__model = CNNDriver()
        # Set model to train mode
        self.__model.train()
        print(self.__model)
        self.__writer.add_graph(self.__model, torch.rand(10, 3, 66, 200))
        # Put model on GPU
        self.__model = self.__model.to(self.__device)

    def train(self, num_epochs=100, batch_size=400, lr=0.0001, l2_norm=0.001, save_dir='./save', input='./DataLMDB'):
        # Create log/save directory if it does not exist
        if not os.path.exists('./logs'):
            os.makedirs('./logs')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.__transformations = transforms.Compose([AugmentDrivingTransform(), DrivingDataToTensor()])
        self.__dataset_train = DriveData_LMDB(input, self.__transformations)
        self.__train_loader = DataLoader(self.__dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

        # Loss and Optimizer
        self.__loss_func = nn.MSELoss()
        # self.__loss_func = nn.SmoothL1Loss()
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr=lr, weight_decay=l2_norm)

        # Decay LR by a factor of 0.1 every 10 epochs
        self.__exp_lr_scheduler = lr_scheduler.StepLR(self.__optimizer, step_size=15, gamma=0.1)

        print('Train size:', len(self.__dataset_train), 'Batch size:', batch_size)
        print('Batches per epoch:', len(self.__dataset_train) // batch_size)

        # Train the Model
        iteration_count = 0
        for epoch in range(num_epochs):
            for batch_idx, samples in enumerate(self.__train_loader):

                # Send inputs/labels to GPU
                images = samples['image'].to(self.__device)
                labels = samples['label'].to(self.__device)

                self.__optimizer.zero_grad()

                # Forward + Backward + Optimize
                outputs = self.__model(images)
                loss = self.__loss_func(outputs, labels.unsqueeze(dim=1))

                loss.backward()
                self.__optimizer.step()
                self.__exp_lr_scheduler.step(epoch)

                # Send loss to tensorboard
                self.__writer.add_scalar('loss/', loss.item(), iteration_count)
                self.__writer.add_histogram('steering_out', outputs.clone().detach().cpu().numpy(), iteration_count, bins='doane')
                self.__writer.add_histogram('steering_in', labels.unsqueeze(dim=1).clone().detach().cpu().numpy(), iteration_count, bins='doane')

                # Get current learning rate (To display on Tensorboard)
                for param_group in self.__optimizer.param_groups:
                    curr_learning_rate = param_group['lr']
                    self.__writer.add_scalar('learning_rate/', curr_learning_rate, iteration_count)

                # Display on each epoch
                if batch_idx == 0:
                    # Send image to tensorboard
                    self.__writer.add_image('Image', images, epoch)
                    self.__writer.add_text('Steering', 'Steering:' + str(outputs[batch_idx].item()), epoch)
                    # Print Epoch and loss
                    print('Epoch [%d/%d] Loss: %.4f' % (epoch + 1, num_epochs, loss.item()))
                    # Save the Trained Model parameters
                    torch.save(self.__model.state_dict(), save_dir+'/cnn_' + str(epoch) + '.pkl')

                iteration_count += 1


if __name__ == '__main__':
    fire.Fire(Train)