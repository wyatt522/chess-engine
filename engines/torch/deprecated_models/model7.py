import torch.nn as nn

# Attempting Squeeze and excitation
class ChessModel7(nn.Module):
    def __init__(self, num_classes):
        super(ChessModel7, self).__init__()
        r = 8
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
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AvgPool2d(8)
        self.sq_fc1 = nn.Linear(64, 8)
        self.ex_fc1 = nn.Linear(8, 64)
        self.sq_fc2 = nn.Linear(64, 8)
        self.ex_fc2 = nn.Linear(8, 64)
        self.sq_fc3 = nn.Linear(64, 8)
        self.ex_fc3 = nn.Linear(8, 64)
        self.sq_fc4 = nn.Linear(64, 8)
        self.ex_fc4 = nn.Linear(8, 64)
        self.sq_fc5 = nn.Linear(64, 8)
        self.ex_fc5 = nn.Linear(8, 64)

        # Initialize weights
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv4.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv5.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv6.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.sq_fc1.weight)
        nn.init.xavier_uniform_(self.ex_fc1.weight)
        nn.init.xavier_uniform_(self.sq_fc2.weight)
        nn.init.xavier_uniform_(self.ex_fc2.weight)
        nn.init.xavier_uniform_(self.sq_fc3.weight)
        nn.init.xavier_uniform_(self.ex_fc3.weight)
        nn.init.xavier_uniform_(self.sq_fc4.weight)
        nn.init.xavier_uniform_(self.ex_fc4.weight)
        nn.init.xavier_uniform_(self.ex_fc5.weight)
        nn.init.xavier_uniform_(self.sq_fc5.weight)

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))

        flat_avg = self.flatten(self.avg_pool(x))
        squeezed = self.relu(self.sq_fc1(flat_avg))
        scale = self.sigmoid(self.ex_fc1(squeezed))
        reshaped_scale = scale.reshape(-1, 64, 1, 1)
        x = x*reshaped_scale
        
        x = self.bn2(self.relu(self.conv2(x)))

        flat_avg = self.flatten(self.avg_pool(x))
        squeezed = self.relu(self.sq_fc2(flat_avg))
        scale = self.sigmoid(self.ex_fc2(squeezed))
        reshaped_scale = scale.reshape(-1, 64, 1, 1)
        x = x*reshaped_scale

        x = self.bn3(self.relu(self.conv3(x)))

        flat_avg = self.flatten(self.avg_pool(x))
        squeezed = self.relu(self.sq_fc3(flat_avg))
        scale = self.sigmoid(self.ex_fc3(squeezed))
        reshaped_scale = scale.reshape(-1, 64, 1, 1)
        x = x*reshaped_scale

        x = self.bn4(self.relu(self.conv4(x)))

        flat_avg = self.flatten(self.avg_pool(x))
        squeezed = self.relu(self.sq_fc4(flat_avg))
        scale = self.sigmoid(self.ex_fc4(squeezed))
        reshaped_scale = scale.reshape(-1, 64, 1, 1)
        x = x*reshaped_scale

        x = self.bn5(self.relu(self.conv5(x)))
        
        flat_avg = self.flatten(self.avg_pool(x))
        squeezed = self.relu(self.sq_fc5(flat_avg))
        scale = self.sigmoid(self.ex_fc5(squeezed))
        reshaped_scale = scale.reshape(-1, 64, 1, 1)
        x = x*reshaped_scale

        x = self.bn6(self.relu(self.conv6(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Output raw logits
        return x
