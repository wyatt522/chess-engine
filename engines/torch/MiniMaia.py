import torch.nn as nn
import torch



class MiniMaiaBlock(nn.Module):
    def __init__(self):
        super(MiniMaiaBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AvgPool2d(8)
        self.sq_fc1 = nn.Linear(64, 8)
        self.ex_fc1 = nn.Linear(8, 64)

        
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')

        nn.init.xavier_uniform_(self.sq_fc1.weight)
        nn.init.xavier_uniform_(self.ex_fc1.weight)
    
    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))

        x2 = self.bn2(self.conv2(x1))

        flat_avg = self.flatten(self.avg_pool(x2))
        squeezed = self.relu(self.sq_fc1(flat_avg))
        scale = self.sigmoid(self.ex_fc1(squeezed))
        reshaped_scale = scale.reshape(-1, 64, 1, 1)

        x_se = x2*reshaped_scale

        x_out = self.relu(x1 + x_se)

        return x_out


# Block based squeeze and excite net
class MiniMaia(nn.Module):
    def __init__(self, num_classes, num_blocks = 6, squeeze_layer = 1024):
        super(MiniMaia, self).__init__()
        # Declare Layers and Blocks
        self.init_layer = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.final_layer = nn.Conv2d(64, 80, kernel_size=3, padding=1)
        self.init_bn = nn.BatchNorm2d(64)
        self.final_bn = nn.BatchNorm2d(80)
        self.fc1 = nn.Linear(64 * 80, squeeze_layer)
        self.fc2 = nn.Linear(squeeze_layer, num_classes)
        
        self.mid_layers_list = []
        for i in range(num_blocks):
            self.mid_layers_list.append(MiniMaiaBlock())

        self.mid_layers = nn.Sequential(*self.mid_layers_list)

        # Needed transformations
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Initialize weights
        nn.init.kaiming_uniform_(self.init_layer.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.final_layer.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.relu(self.init_bn(self.init_layer(x)))

        x = self.mid_layers(x)
        
        x = self.relu(self.final_bn(self.final_layer(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Output raw logits
        return x

# Block based squeeze and excite net
class MiniMaiaSkip(nn.Module):
    def __init__(self, num_classes, num_blocks = 6, squeeze_layer = 1024):
        super(MiniMaiaSkip, self).__init__()
        # Declare Layers and Blocks
        self.init_layer = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.final_layer = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.init_bn = nn.BatchNorm2d(64)
        self.final_bn = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 80, squeeze_layer)
        self.fc2 = nn.Linear(squeeze_layer, num_classes)
        
        self.mid_layers_list = []
        for i in range(num_blocks):
            self.mid_layers_list.append(MiniMaiaBlock())

        self.mid_layers = nn.Sequential(*self.mid_layers_list)

        # Needed transformations
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Initialize weights
        nn.init.kaiming_uniform_(self.init_layer.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.final_layer.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x1 = self.relu(self.init_bn(self.init_layer(x)))

        x2 = self.mid_layers(x1)
        
        x3 = self.relu(self.final_bn(self.final_layer(x2)))
        x4 = torch.cat([x3,x], 1)
        x5 = self.flatten(x4)
        x6 = self.relu(self.fc1(x5))
        x7 = self.fc2(x6)  # Output raw logits
        return x7


# Block based squeeze and excite net without fully connected final layers
class MiniMaiaSkipFC(nn.Module):
    def __init__(self, num_classes, num_blocks = 6):
        super(MiniMaiaSkipFC, self).__init__()
        # Declare Layers and Blocks
        self.init_layer = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.final_layer = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.init_bn = nn.BatchNorm2d(64)
        self.final_bn = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64 * 80, num_classes)
        
        self.mid_layers_list = []
        for i in range(num_blocks):
            self.mid_layers_list.append(MiniMaiaBlock())

        self.mid_layers = nn.Sequential(*self.mid_layers_list)

        # Needed transformations
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Initialize weights
        nn.init.kaiming_uniform_(self.init_layer.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.final_layer.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x1 = self.relu(self.init_bn(self.init_layer(x)))

        x2 = self.mid_layers(x1)
        
        x3 = self.relu(self.final_bn(self.final_layer(x2)))
        x4 = torch.cat([x3,x], 1)
        x5 = self.flatten(x4)
        x6 = self.fc(x5)
        return x6