import torch
from torch.nn import functional as F

class ActorCritic(torch.nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        # Assuming input images are 210x160x3 (Atari game dimensions)
        
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4, stride=1)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=1)
        self.pool5 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = torch.nn.Linear(in_features=192, out_features=512)
        self.fc2 = torch.nn.Linear(in_features=512, out_features=512)

        self.actor_lin = torch.nn.Linear(in_features=512, out_features=2)  # returns 2 possible actions
        self.critic_lin = torch.nn.Linear(in_features=512, out_features=1)  # returns a single number

    def forward(self, x):
      
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
    
        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        actor = F.log_softmax(self.actor_lin(x), dim=1)
        actor = torch.exp(actor)
        
        critic = torch.tanh(self.critic_lin(x))
        
        return actor, critic

    def _conv_output_size(self, size, kernel_size, stride):
        return ((size[0] - (kernel_size - 1) - 1) // stride + 1,
                (size[1] - (kernel_size - 1) - 1) // stride + 1)