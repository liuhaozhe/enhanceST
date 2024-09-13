import torch
import torch.nn as nn
import torch.optim as optim
from .configs import Config
from .utils import *
import time

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out += x
        out = self.relu(out)
        return out

class Net(nn.Module):
    def __init__(self, train_set, val_set, test_set, conf=Config):
        super(Net, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conf = conf

        self.train_lr = []
        self.train_hr = []
        self.train_in_tissue = []
        self.loss = []

        self.val_lr = []
        self.val_hr = []
        self.val_in_tissue = []
        self.val_loss = []

        if isinstance(train_set[0], list):
            print("Multiple Train data")
            for i in range(len(train_set)):
                self.train_lr.append(train_set[i][0])
                self.train_hr.append(train_set[i][1])
                self.train_in_tissue.append(train_set[i][2])
        else:
            print("Single Train data")
            self.train_lr.append(train_set[0])
            self.train_hr.append(train_set[1])
            self.train_in_tissue.append(train_set[2])

        if isinstance(val_set[0], list):
            print("Multiple Validation data")
            for i in range(len(val_set)):
                self.val_lr.append(val_set[i][0])
                self.val_hr.append(val_set[i][1])
                self.val_in_tissue.append(val_set[i][2])
        else:
            print("Single Validation data")
            self.val_lr.append(val_set[0])
            self.val_hr.append(val_set[1])
            self.val_in_tissue.append(val_set[2])

        self.test_set = test_set

        self.build_network(conf)

        self.to(self.device)

        self.optimizer = optim.Adam([
            {'params': self.net_params_1, 'lr': conf.learning_rate},
            {'params': self.net_params_2, 'lr': conf.learning_rate * 0.001}
        ])

        self.criterion = nn.MSELoss(reduction='none')


    def build_network(self, meta):
        self.layers = nn.ModuleList()

        for ind in range(meta.depth):
            if (ind + 1) % meta.iteration_depth == 0:
                in_channels = meta.filter_shape[ind][2] if ind == 0 else meta.filter_shape[ind - 1][-1]
                out_channels = meta.filter_shape[ind][-1]
                block = ResidualBlock(in_channels, out_channels, kernel_size=meta.filter_shape[ind][0:2])
                self.layers.append(block)
            else:
                conv_layer = nn.Conv2d(
                    in_channels=meta.filter_shape[ind][2] if ind == 0 else meta.filter_shape[ind - 1][-1],
                    out_channels=meta.filter_shape[ind][-1],
                    kernel_size=meta.filter_shape[ind][0:2],
                    padding='same'
                )
                self.layers.append(conv_layer)
                self.layers.append(nn.ReLU())

        self.net_params_1 = list(self.layers[:-1].parameters())
        self.net_params_2 = list(self.layers[-1:].parameters())

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        return x

    def run(self):

        losses_list = []
        val_losses_list=[]
        losses = 0
        val_losses = 0
        t = self.conf.epoch
        for epoch in range(t):
            start = time.time()

            self.train_lr_batch, self.train_hr_batch, self.train_in_tissue_batch = minibatch_list(self.train_lr,
                                                                                                  self.train_hr,
                                                                                                  self.train_in_tissue,
                                                                                                  batch_size=self.conf.batch_size,
                                                                                                  seed=42)
            # validation
            self.val_lr_batch, self.val_hr_batch, self.val_in_tissue_batch = minibatch_list(self.val_lr,
                                                                                            self.val_hr,
                                                                                            self.val_in_tissue,
                                                                                            batch_size=self.conf.batch_size,
                                                                                            seed=42)


            for self.input in zip(self.train_lr_batch, self.train_hr_batch, self.train_in_tissue_batch):
                loss = self.train_step()
                losses += loss
            losses /= len(self.train_lr_batch)
            losses_list.append(losses)


            for self.input in zip(self.val_lr_batch, self.val_hr_batch, self.val_in_tissue_batch):
                val_loss = self.evaluate()
                val_losses += val_loss
            val_losses /= len(self.val_lr_batch)
            val_losses_list.append(val_losses)


            i = epoch + 1
            end = time.time()
            time_per_epoch = end - start
            print(f'Epoch {i}/{t}, loss: {losses:.4f}, val_loss: {val_losses:.4f}, time_per_epoch: {time_per_epoch:.4f}')


        t = len(self.test_set)
        test_output = np.zeros([t, self.test_set[0].shape[0] * 2, self.test_set[0].shape[1] * 2])
        # print(test_output.shape)
        for g, self.input in enumerate(self.test_set):
            self.input = np.expand_dims(self.input, axis=0)

            output = self.forward_pass(self.input)

            test_output[g] = output

        return test_output, losses_list, val_losses_list

    def forward_pass(self, lr):
        lr_tensor = torch.tensor(lr, dtype=torch.float32).unsqueeze(0).to(self.device)

        output = self.forward(lr_tensor)

        output = output.detach().cpu().numpy().squeeze()

        if self.conf.test_positive:
            output[output < 0] = 0

        output = np.transpose(output, (1, 2, 0))
        up_output = np.zeros([output.shape[0] * 2, output.shape[1] * 2])

        up_output[0::2, 0::2] = output[:, :, 0]
        up_output[0::2, 1::2] = output[:, :, 1]
        up_output[1::2, 0::2] = output[:, :, 2]
        up_output[1::2, 1::2] = output[:, :, 3]
        return up_output

    def train(self, mode: bool = True):
        super().train(mode)

    def train_step(self):
        self.train(True)

        self.lr = torch.tensor(np.array(self.input[0]), dtype=torch.float32).unsqueeze(1).to(self.device)
        self.hr = torch.tensor(np.array(self.input[1]), dtype=torch.float32).unsqueeze(1).to(self.device)
        self.in_tissue = torch.tensor(self.input[2], dtype=torch.float32).unsqueeze(0).to(self.device)


        self.optimizer.zero_grad()

        output = self.forward(self.lr)

        loss_0 = (self.criterion(output[:, 0, :, :], self.hr[:, 0, 0::2, 0::2]) * self.in_tissue[:, 0::2, 0::2]).view(-1)
        loss_1 = (self.criterion(output[:, 1, :, :], self.hr[:, 0, 0::2, 1::2]) * self.in_tissue[:, 0::2, 1::2]).view(-1)
        loss_2 = (self.criterion(output[:, 2, :, :], self.hr[:, 0, 1::2, 0::2]) * self.in_tissue[:, 1::2, 0::2]).view(-1)
        loss_3 = (self.criterion(output[:, 3, :, :], self.hr[:, 0, 1::2, 1::2]) * self.in_tissue[:, 1::2, 1::2]).view(-1)

        losses_concat = torch.cat([loss_0, loss_1, loss_2, loss_3], dim=0)

        total_loss = losses_concat.mean()

        total_loss.backward()
        self.optimizer.step()

        self.loss.append(total_loss.item())

        return total_loss.item()

    def evaluate(self):
        self.eval()
        with torch.no_grad():

            self.lr = torch.tensor(np.array(self.input[0]), dtype=torch.float32).unsqueeze(1).to(self.device)
            self.hr = torch.tensor(np.array(self.input[1]), dtype=torch.float32).unsqueeze(1).to(self.device)
            self.in_tissue = torch.tensor(self.input[2], dtype=torch.float32).unsqueeze(0).to(self.device)


            output = self.forward(self.lr)

            loss_0 = (self.criterion(output[:, 0, :, :], self.hr[:, 0, 0::2, 0::2]) * self.in_tissue[:, 0::2, 0::2]).view(-1)
            loss_1 = (self.criterion(output[:, 1, :, :], self.hr[:, 0, 0::2, 1::2]) * self.in_tissue[:, 0::2, 1::2]).view(-1)
            loss_2 = (self.criterion(output[:, 2, :, :], self.hr[:, 0, 1::2, 0::2]) * self.in_tissue[:, 1::2, 0::2]).view(-1)
            loss_3 = (self.criterion(output[:, 3, :, :], self.hr[:, 0, 1::2, 1::2]) * self.in_tissue[:, 1::2, 1::2]).view(-1)

            val_losses_concat = torch.cat([loss_0, loss_1, loss_2, loss_3], dim=0)

            val_total_loss = val_losses_concat.mean()

        self.val_loss.append(val_total_loss.item())

        return val_total_loss.item()