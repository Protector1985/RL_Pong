import torch
from torch.nn import functional as F

class ActorCritic(torch.nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        # Assuming input images are 210x160x3 (Atari game dimensions)
        
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute the size of the flattened features after convolutional layers
        convw = self._conv_output_size((210, 160), 8, 4)
        convw = self._conv_output_size(convw, 4, 2)
        convw = self._conv_output_size(convw, 3, 1)
        linear_input_size = convw[0] * convw[1] * 64

        self.fc1 = torch.nn.Linear(linear_input_size, 512)

        self.actor_lin = torch.nn.Linear(512, 2)  # returns 2 possible actions
        self.critic_lin = torch.nn.Linear(512, 1)  # returns a single number

    def forward(self, x):
      
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
    
        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = F.relu(self.fc1(x))

        actor = F.log_softmax(self.actor_lin(x), dim=1)
        critic = torch.tanh(self.critic_lin(x))

        return actor, critic

    def _conv_output_size(self, size, kernel_size, stride):
        return ((size[0] - (kernel_size - 1) - 1) // stride + 1,
                (size[1] - (kernel_size - 1) - 1) // stride + 1)