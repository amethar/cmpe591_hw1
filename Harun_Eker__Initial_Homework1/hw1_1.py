import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse


import environment


class Hw1Env(environment.BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.obj_type = None  # Initialize obj_type as an instance variable to be able to access it in state

    def _create_scene(self, seed=None): #create scene, object
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        self.obj_type = np.random.rand() #objct radius
        if self.obj_type < 0.5: #box
            size = np.random.uniform([0.02, 0.02, 0.02], [0.03, 0.03, 0.03])
            environment.create_object(scene, "box", pos=[0.6, 0., 1.1], quat=[0, 0, 0, 1],
                                      size=size, rgba=[0.8, 0.2, 0.2, 1], friction=[0.02, 0.005, 0.0001],
                                      density=4000, name="obj1")
        else: #sphere
            size = np.random.uniform([0.02, 0.02, 0.02], [0.03, 0.03, 0.03])
            environment.create_object(scene, "sphere", pos=[0.6, 0., 1.1], quat=[0, 0, 0, 1],
                                      size=size, rgba=[0.8, 0.2, 0.2, 1], friction=[0.2, 0.005, 0.0001],
                                      density=4000, name="obj1")
        return scene

    def state(self): #state: object pose and pixels in scene
        obj_pos = self.data.body("obj1").xpos[:2]

        if self.obj_type < 0.5:
            obj_type = 0
        else:
            obj_type = 1
        #if self._render_mode == "offscreen":
        #    self.viewer.update_scene(self.data, camera="topdown") #no need for the images
        #    pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1) #no need for the images
        #else:
        #    pixels = self.viewer.read_pixels(camid=1).copy() #no need for the images
        #    pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1) #no need for the images
        #    pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:])) #no need for the images
        #    pixels = transforms.functional.resize(pixels, (128, 128)) #no need for the images
        return obj_pos, obj_type#, pixels #no need for the images

    def step(self, action_id): #take one of four actions, return nothing. blocking?
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


def collect_data(idx, N):

    env = Hw1Env(render_mode="offscreen") # gui or offscreen

    env.reset() #to make sure everthing is initalised. I get an error if I do not call reset() before calling state()


    obj_types = torch.zeros(N, dtype=torch.uint8) #1d object type
    actions = torch.zeros(N, 4, dtype=torch.uint8) #4d actions

    state_action = torch.zeros(N, 5, dtype=torch.uint8) #to be 5d neural network intput

    positions = torch.zeros(N, 2, dtype=torch.float) #2d reference output
    #imgs = torch.zeros(N, 3, 128, 128, dtype=torch.uint8) #no need for the images

    for i in range(N):
        print(f"Simulating {i + 1} out of {N}")

        action_id = np.random.randint(4)
        
        env.step(action_id)
        #obj_pos, pixels = env.state()  #no need for the images

        obj_pos, obj_type = env.state()

        positions[i] = torch.tensor(obj_pos)
        obj_types[i] = torch.tensor(obj_type)
        actions[i][action_id]= 1


        #collecting data to store together
        state_action[i, 0] = obj_types[i]
        state_action[i, 1:] = actions[i]

        env.reset()

    torch.save(state_action, f"state_actions_{idx}.pt")
    torch.save(positions, f"positions_{idx}.pt")

    #torch.save(imgs, f"imgs_{idx}.pt")  #no need for the images


class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.hidden = nn.Linear(5, 4)  # Hidden layer with 4 nodes (input size = 5)
        self.output = nn.Linear(4, 2)  # Output layer (output size = 2)
        self.activation = nn.ReLU()   # Activation function (ReLU for hidden layer)



    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x



def train(idx):
    learning_rate = 0.01
    num_epochs = 1000
    train_ratio = 0.8


    model = MLPModel()
    criterion = nn.MSELoss()  # mean squared error based Loss for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam for step size calculation
 
    inputs = torch.load(f"state_actions_{idx}.pt").float() # linear layer requires Float tensors
    reference_outputs = torch.load(f"positions_{idx}.pt")

    num_samples = inputs.size(0)
    num_train = int(num_samples * train_ratio)
    num_test = num_samples - num_train

    # shuffle the indices and split the dataset
    indices = torch.randperm(num_samples)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    # split the inputs and reference outputs
    train_inputs = inputs[train_indices]
    train_reference_outputs = reference_outputs[train_indices]

    test_inputs = inputs[test_indices]
    test_reference_outputs = reference_outputs[test_indices]

    #to store losses
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
    
        # forward processing
        train_outputs = model(train_inputs) #neural network output
        train_loss = criterion(train_outputs, train_reference_outputs) #calculating loss based on output (predicted object pose) and reference output
    
        # backward pass and optimization
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step() #updating parameters of the network
    
        #run on test dataset
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_inputs)
            test_loss = criterion(test_outputs, test_reference_outputs)

        # store the losses
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

        # print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

    #plot the losses for training and testing over epochs
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Losses')
    plt.legend()
    plt.show()        

    torch.save(model, "mlp_model.pth") #save the model




if __name__ == "__main__":
    #in order not to rerun the simulation to collect data, data collection is optional. If data is properly stored, training can use it with no need to run the simulation.
    #to run data collection, run: python homework1.py --collect_data
    #to train with existin data, run: python homework1.py

    #argparse is used to handle arguments
    parser = argparse.ArgumentParser(description="Run training with optional data collection.")
    parser.add_argument("--collect_data", action="store_true", help="Run the data collection step before training.")
    args = parser.parse_args()

    number_of_sims = 50
    dataset_id = number_of_sims

    if args.collect_data:
        print("Running data collection...")
        #multiprocessing does not work in my computer (XIO:  fatal IO error 0 (Success) on X server ":0" after 171 requests (171 known processed) with 0 events remaining.)
        #so, working in a single process
        collect_data(dataset_id, number_of_sims)

    print("Starting training...")
    train(dataset_id)


