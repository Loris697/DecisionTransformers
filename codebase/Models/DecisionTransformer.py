import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Callable

from ..ModelTester import ModelTester  # Ensure this import path is correct in your project structure.

def merge_embedding_sequence(reward, observation, action):
    """
    Merge embeddings from separate tensors into a single sequence tensor.

    Args:
        reward (torch.Tensor): Embedding for rewards.
        observation (torch.Tensor): Embedding for observations.
        action (torch.Tensor): Embedding for actions.

    Returns:
        torch.Tensor: Merged embedding tensor.
    """
    batch_size, seq_len, emb_dim = reward.size()
    stacked = torch.stack((reward, observation, action), dim=0)
    permuted = stacked.permute(1, 2, 0, 3)  # New shape: [batch_size, seq_len, 3, emb_dim]
    result = permuted.reshape(batch_size, seq_len * 3, emb_dim)
    return result

def set_elements_to_zero(row, index):
    """
    Sets the elements of a tensor row to zero starting from a given index.

    Args:
        row (torch.Tensor): The tensor row where elements will be set to zero.
        index (int): The start index from where to set elements to zero.

    Returns:
        torch.Tensor: Modified tensor row.
    """
    mask = torch.zeros_like(row)
    mask[index:] = 1
    row = row * mask
    return torch.nan_to_num(row, neginf=-float('inf'))

def step_masking(seq_len):
    """
    Create a step masking tensor for the attention mechanism to only allow
    attention to previous and current steps in a sequence.

    Args:
        seq_len (int): The total length of the sequence.

    Returns:
        torch.Tensor: The attention mask tensor.
    """
    step_len = seq_len // 3
    attention_mask = torch.full((seq_len, seq_len), -float('inf'), dtype=torch.float32)
    for step in range(1, step_len + 1):
        row_set = step - 1
        for sequence_element in [row_set * 3, row_set * 3 + 1, row_set * 3 + 2]:
            attention_mask[sequence_element] = set_elements_to_zero(attention_mask[sequence_element], step * 3)
    
    return attention_mask

def repeat_integers(integers):
    """
    Repeat each integer in a tensor three times.

    Args:
        integers (torch.Tensor): Tensor of integers.

    Returns:
        torch.Tensor: Tensor with repeated integers.
    """
    return torch.repeat_interleave(integers, 3)

class DecisionTransformers(pl.LightningModule):
    def __init__(self, embedding_reward: nn.Module, embedding_action: nn.Module,
                 embedding_observation: nn.Module, transformer: nn.Module,
                 optimizer: Callable, action_space_dim: int, loss: Callable = F.huber_loss,
                 lr: float = 1e-4, d_model: int = 128, modelTester: ModelTester = None):
        """
        Decision transformer module that uses separate embeddings for rewards, actions,
        and observations, and processes these via a transformer architecture.

        Args:
            embedding_reward (nn.Module): Embedding module for rewards.
            embedding_action (nn.Module): Embedding module for actions.
            embedding_observation (nn.Module): Embedding module for observations.
            transformer (nn.Module): Transformer module for processing sequences.
            optimizer (Callable): Optimizer class to be used for training.
            action_space_dim (int): Dimension of the action space.
            loss (Callable, optional): Loss function, default is Huber loss.
            lr (float): Learning rate for the optimizer.
            d_model (int): Dimension of the embeddings.
            modelTester (ModelTester, optional): Tester module for evaluating the model during training.
        """
        super(DecisionTransformers, self).__init__()
        self.embedding_reward = embedding_reward
        self.embedding_action = embedding_action
        self.embedding_observation = embedding_observation
        self.transformer = transformer
        transformer.setPosition(repeat_integers(torch.arange(0, transformer.max_step_len // 3)))
        
        self.optimizer_callable = optimizer
        self.loss = loss
        self.lr = lr
        self.fc1 = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU())
        self.output = nn.Linear(2 * d_model, action_space_dim)
        self.action_space_dim = action_space_dim
        self.modelTester = modelTester
        self.epoch = 0
        
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Number of learnable parameters:", num_params)

    def forward(self, x):
        """
        Forward pass through the decision transformer.

        Args:
            x (dict): Dictionary containing 'rewards', 'observations', and 'actions'.

        Returns:
            torch.Tensor: Output from the model.
        """
        rewards_emb = self.embedding_reward(x["rewards"])
        actions_emb = self.embedding_action(x["actions"])
        observations_emb = self.embedding_observation(x["observations"].reshape(-1, *x["observations"].shape[2:]))
        observations_emb = observations_emb.view(x["observations"].shape[0], x["observations"].shape[1], -1)
        
        sequence = merge_embedding_sequence(rewards_emb, observations_emb, actions_emb)
        
        attention_mask = step_masking(self.transformer.max_step_len).to(x["observations"].device)
        output = self.transformer(sequence, attention_mask=attention_mask)
        output = output[:, 1::3, :]  # Extract the output related to observations
        output = self.fc1(output)
        return self.output(output)

    def training_step(self, batch, batch_idx):
        """
        Training step for the model using provided batch.

        Args:
            batch (tuple): Input and target data for the training.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        x, y = batch
        y_hat = self(x)
        total_loss = 0
        for dim in range(self.action_space_dim):
            loss = self.loss(y_hat[:, :, dim], y[:, :, dim])
            total_loss += loss
            self.log(f'train_loss_dim_{dim}', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def on_train_epoch_end(self):
        """
        Actions to perform at the end of each training epoch.
        """
        self.epoch += 1
        self.eval()
        if self.epoch % 400 == 0 and self.modelTester:
            test_results = self.modelTester.test_model(
                self,
                episodes=3, 
                starting_rewards=[0., 0.5, 1., 1.2],
                render=True, folder=f"output_folder/epoch_{self.epoch}/"
            )
            for starting_reward in test_results:
                self.log(f'starting_reward_{starting_reward}', 
                         test_results[starting_reward], 
                         on_epoch=True, prog_bar=True)
            print(f"Test results: {test_results}")
        self.train()
    
    def configure_optimizers(self):
        """
        Configure the optimizer for the model.

        Returns:
            torch.optim.Optimizer: Optimizer for the model.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)