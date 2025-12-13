import torch.nn as nn

# no FCNN, just weird cnn 
class ChessModel5(nn.Module):
    def __init__(self, num_classes):
        super(ChessModel5, self).__init__()
        # conv1 -> relu -> conv2 -> relu -> conv3 -> relu -> conv4 -> relu -> flatten -> fc1 -> relu -> fc2
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 80, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(80, num_classes, kernel_size=8)
        self.flatten = nn.Flatten()
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(80)
        self.relu = nn.ReLU()

        # Initialize weights
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv4.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv5.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv6.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv7.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))
        x = self.bn4(self.relu(self.conv4(x)))
        x = self.bn5(self.relu(self.conv5(x)))
        x = self.bn6(self.relu(self.conv6(x)))
        x = self.conv7(x)
        x = self.flatten(x)
        return x
