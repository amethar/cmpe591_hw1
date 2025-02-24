import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import environment


class Hw1Env(environment.BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        r = np.random.rand()
        if r < 0.5:
            size = np.random.uniform([0.02, 0.02, 0.02], [0.03, 0.03, 0.03])
            environment.create_object(scene, "box", pos=[0.6, 0., 1.1], quat=[0, 0, 0, 1],
                                      size=size, rgba=[0.8, 0.2, 0.2, 1], friction=[0.02, 0.005, 0.0001],
                                      density=4000, name="obj1")
        else:
            size = np.random.uniform([0.02, 0.02, 0.02], [0.03, 0.03, 0.03])
            environment.create_object(scene, "sphere", pos=[0.6, 0., 1.1], quat=[0, 0, 0, 1],
                                      size=size, rgba=[0.8, 0.2, 0.2, 1], friction=[0.2, 0.005, 0.0001],
                                      density=4000, name="obj1")
        return scene

    def state(self):
        obj_pos = self.data.body("obj1").xpos[:2]
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="topdown")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=1).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return obj_pos, pixels

    def step(self, action_id):
        if action_id == 0:
            self._set_joint_position({6: 0.8})
            self._set_ee_in_cartesian([0.4, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.8, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.4, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})
        elif action_id == 1:
            self._set_joint_position({6: 0.8})
            self._set_ee_in_cartesian([0.8, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.4, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.8, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})
        elif action_id == 2:
            self._set_joint_position({6: 0.8})
            self._set_ee_in_cartesian([0.6, -0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.6, 0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.6, -0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})
        elif action_id == 3:
            self._set_joint_position({6: 0.8})
            self._set_ee_in_cartesian([0.6, 0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.6, -0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.6, 0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})


def collect(train_dataset_size, test_dataset_size):

    env = Hw1Env(render_mode="offscreen")

    #collect train data

    positions = torch.zeros(train_dataset_size, 2, dtype=torch.float) 
    actions = torch.zeros(train_dataset_size, dtype=torch.uint8)
    imgs_before = torch.zeros(train_dataset_size, 3, 128, 128, dtype=torch.uint8)
    #imgs_after = torch.zeros(train_dataset_size, 3, 128, 128, dtype=torch.uint8)

    for i in range(train_dataset_size):
        print(f"Simulating {i + 1} out of {train_dataset_size} for training dataset")
        env.reset()
        action_id = np.random.randint(4)
        obj_pos, img_before = env.state() #initial position not used
        imgs_before[i] = img_before
        env.step(action_id)
        obj_pos, img_after = env.state()
        positions[i] = torch.tensor(obj_pos) 
        actions[i] = action_id
        #imgs_after[i] = img_after
    
    torch.save(positions, "p2_train_dataset_reference_outputs.pt") 
    torch.save(actions, "p2_train_dataset_actions.pt")
    torch.save(imgs_before, "p2_train_dataset_imgs_before.pt")
    #torch.save(imgs_after, "p2_train_dataset_imgs_after.pt") #final images are not used



    #collect test data
    positions = torch.zeros(test_dataset_size, 2, dtype=torch.float) 
    actions = torch.zeros(test_dataset_size, dtype=torch.uint8)
    imgs_before = torch.zeros(test_dataset_size, 3, 128, 128, dtype=torch.uint8)
    #imgs_after = torch.zeros(test_dataset_size, 3, 128, 128, dtype=torch.uint8)

    for i in range(test_dataset_size):
        print(f"Simulating {i + 1} out of {test_dataset_size} for test dataset")
        env.reset()
        action_id = np.random.randint(4)
        obj_pos, img_before = env.state() #initial position not used
        imgs_before[i] = img_before
        env.step(action_id)
        obj_pos, img_after = env.state()
        positions[i] = torch.tensor(obj_pos) 
        actions[i] = action_id
        #imgs_after[i] = img_after
    
    torch.save(positions, "p2_test_dataset_reference_outputs.pt") 
    torch.save(actions, "p2_test_dataset_actions.pt")
    torch.save(imgs_before, "p2_test_dataset_imgs_before.pt")
    #torch.save(imgs_after, "p2_test_dataset_imgs_after.pt") #final images are not used



class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # convolutional layers 
        self.conv_layers = nn.Sequential(
            #input size: # 128x138x3

            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),  #resulting size 128x138x8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #resulting size 64x64x8

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1), #resulting size 64x64x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #resulting size 32x32x16

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), #resulting size 32x32x8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #resulting size 16x16x32
        )

        # fully connected layers for combining image and action
        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 16 * 32 + 4, 64),  # input size: 16x16x32 + 4 actions (one-hot encoding), output size: 64
            nn.ReLU(),
            nn.Linear(64, 2), #input size: 64, output size: 2 (x,y position)
        )

    def forward(self, img, action):
        # forward CNN layers
        x = self.conv_layers(img)
        x = x.view(x.size(0), -1)  # flatten output of cnn

        # concatenate cnn output with action (one-hot encoded)
        x = torch.cat((x, action), dim=1)

        # Fully connected layers to predict position
        x = self.fc_layers(x)
        return x



def train():
    learning_rate = 0.001
    num_epochs = 1000
    train_ratio = 0.8


    model = CNNModel()
    criterion = nn.MSELoss()   # mean squared error based loss for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam for step calculation


    images = torch.load("p2_train_dataset_imgs_before.pt").float() #convolutional layer requires float tensors
    images = images / 255.0 # I constantly get normalization suggestions, but not sure about the reason
    actions = torch.load("p2_train_dataset_actions.pt")
    reference_outputs = torch.load("p2_train_dataset_reference_outputs.pt")


    num_samples = images.size(0)
    num_train = int(num_samples * train_ratio)
    num_test = num_samples - num_train

    # shuffle the indices and split the dataset
    indices = torch.randperm(num_samples)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    # split the inputs and reference outputs
    train_images = images[train_indices]
    train_actions = actions[train_indices]
    train_reference_outputs = reference_outputs[train_indices]

    train_actions = train_actions.view(-1).to(torch.long)   # making it 1d
    train_actions_one_hot = torch.zeros(num_train, 4) #will be passed to the network as one-hot
    train_actions_one_hot[torch.arange(num_train), train_actions] = 1


    test_images = images[test_indices]
    test_actions = actions[test_indices]
    test_reference_outputs = reference_outputs[test_indices]

    test_actions = test_actions.view(-1).to(torch.long)  # making it 1d
    test_actions_one_hot = torch.zeros(num_test, 4)
    test_actions_one_hot[torch.arange(num_test), test_actions] = 1


    train_losses = []
    test_losses = []


    for epoch in range(num_epochs):
        model.train()

        train_outputs = model(train_images, train_actions_one_hot) #neural network output
        train_loss = criterion(train_outputs, train_reference_outputs) #calculating loss based on output (predicted object pose) and reference output

        # backward pass and optimization
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step() #updating parameters of the network

        #run on test dataset
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_images, test_actions_one_hot)
            test_loss = criterion(test_outputs, test_reference_outputs)
        
        # store the losses
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

        # print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

    #plot the losses for training and validation over epochs
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Losses')
    plt.legend()
    plt.show() 

    #save the model
    torch.save(model, "hw1_2.pt")


def test():
    model = torch.load("hw1_2.pt", weights_only=False) #load the model
    model.eval()  # set to evaluation mode

    # load the test dataset
    images = torch.load("p2_test_dataset_imgs_before.pt").float() #convolutional layer requires float tensors
    images = images / 255.0 # I constantly get normalization suggestions, but not sure about the reason
    actions = torch.load("p2_test_dataset_actions.pt")
    reference_outputs = torch.load("p2_test_dataset_reference_outputs.pt")


    num_samples = images.size(0)

    actions = actions.view(-1).to(torch.long)  # making it 1d
    actions_one_hot = torch.zeros(num_samples, 4) #will be passed to the network as one-hot
    actions_one_hot[torch.arange(num_samples), actions] = 1

    # Run the model on the test data
    with torch.no_grad():
        predictions = model(images, actions_one_hot)

    # Compute Mean Squared Error (MSE)
    mse_loss = nn.MSELoss()
    test_loss = mse_loss(predictions, reference_outputs)

    print(f"Test Loss (MSE): {test_loss.item():.4f}")



if __name__ == "__main__":

    train_dataset_size = 1000
    test_dataset_size = 200

    print("Starting data collection...")
    #collect(train_dataset_size, test_dataset_size)

    print("Starting training...")
    #train()

    print("Starting testing...")
    test()

