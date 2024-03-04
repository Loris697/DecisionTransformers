import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .Resnet import CustomResNet
from .Transformers import TransformerArchitecture

def merge_embedding_sequence(reward, observation, action):
    batch_size, seq_len, emb_dim = reward.size()
    combined_tensor = torch.stack((reward, observation, action), dim=1)
    #print(combined_tensor.shape)
    combined_tensor = combined_tensor.view(batch_size, seq_len * 3, emb_dim)
    #print(combined_tensor.shape)
    return combined_tensor

class DecisionTransformers(pl.LightningModule):
    def __init__(self, d_model, action_space_dim, observation_space, batch_first = True, max_seq_len = 32):
        super(DecisionTransformers, self).__init__()

        #reward, action, observation to embedding
        self.embedding_reward = nn.Linear(1, d_model)
        self.embedding_action = nn.Linear(action_space_dim, d_model)
        self.embedding_observation = CustomResNet(observation_space, features_dim = d_model)

        #Defining the Transformer architecture
        self.emb_dim = d_model
        self.transformer = TransformerArchitecture(d_model = d_model, max_step_len=max_seq_len*3, batch_first = batch_first)

        #Defining fully connected layer
        self.fc1 = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
        )
        self.output = nn.Linear(2 * d_model,  action_space_dim)

        self.huber_loss = nn.SmoothL1Loss()  # Huber loss

        # Print the number of learnable parameters
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Number of learnable parameters for the entire architecture:", num_params)
        
    def forward(self, x):
        rewards = x["rewards"]
        observations = x["observations"]
        actions = x["actions"]

        #calculating variable needed after
        batch_len = observations.shape[0]
        seq_len = observations.shape[1]
        device = observations.device

        #Calculating embedding
        rewards_emb = self.embedding_reward(rewards)
        actions_emb = self.embedding_action(actions)
        observations_emb = torch.empty((batch_len,seq_len, self.emb_dim))
        
        for batch_index, batch_imgs in enumerate(observations):
            observations_emb[batch_index] = self.embedding_observation(batch_imgs)

        observations_emb = observations_emb.to(device)
        
        #interleave the sequences
        sequence = merge_embedding_sequence(rewards_emb, observations_emb, actions_emb)

        output = self.transformer(sequence)

        # Extract the output related to the observation input
        sequence = sequence[:, 1::3, :]

        #Fully connected layer to get the action
        output = self.fc1(output)
        
        return self.output(output)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Compute three different dimension of action space
        loss1 = F.huber_loss(y_hat[:, 0], y[:, 0])
        loss2 = F.huber_loss(y_hat[:, 1], y[:, 1])
        loss3 = F.huber_loss(y_hat[:, 2], y[:, 2])
        
        # Total loss is the sum of the three losses
        total_loss = loss1 + loss2 + loss3
        
        self.log('train_loss1', loss1, prog_bar=True)
        self.log('train_loss2', loss2, prog_bar=True)
        self.log('train_loss3', loss3, prog_bar=True)
        
        self.log('train_loss', total_loss, prog_bar=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Compute three different dimension of action space
        loss1 = F.huber_loss(y_hat[:, 0], y[:, 0])  # Loss for first value
        loss2 = F.huber_loss(y_hat[:, 1], y[:, 1])  # Loss for second value
        loss3 = F.huber_loss(y_hat[:, 2], y[:, 2])  # Loss for third value
        
        # Total loss is the sum of the three losses
        total_loss = loss1 + loss2 + loss3
        
        self.log('val_loss1', loss1, prog_bar=True)
        self.log('val_loss2', loss2, prog_bar=True)
        self.log('val_loss3', loss3, prog_bar=True)
        
        self.log('val_loss', total_loss, prog_bar=True)
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer