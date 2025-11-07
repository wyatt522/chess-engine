import torch.nn as nn

# Skip Connection Network
class ChessModel8(nn.Module):
    def __init__(self, num_classes):
        super(ChessModel8, self).__init__()
        # conv1 -> relu -> conv2 -> relu -> conv3 -> relu -> conv4 -> relu -> flatten -> fc1 -> relu -> fc2
        self.conv1 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 80, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(80)
        self.fc1 = nn.Linear(8 * 8 * 80, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

        # Initialize weights
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv4.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv5.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv6.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x1 = self.bn1(self.relu(self.conv1(x)))
        x2 = self.bn2(self.relu(self.conv2(x1)))
        x3 = self.bn3(self.relu(self.conv3(x2)))
        x4 = self.bn4(self.relu(self.conv4(x3 + x1)))
        x5 = self.bn5(self.relu(self.conv5(x4)))
        x6 = self.bn6(self.relu(self.conv6(x5 + x3)))
        x7 = self.flatten(x6)
        x8 = self.relu(self.fc1(x7))
        x9 = self.fc2(x8)  # Output raw logits
        return x9
