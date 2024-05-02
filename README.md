#  Unofficial Decision Transformer  🤖

This repository is dedicated to a reimplementation of the Decision Transformer. Here some example of the same model but with different target rewards.

 Minimum Reward        |  Half Reward  |  Maximum Reward
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/Loris697/DecisionTransformers/blob/main/output_folder/starting_rewards0.0_0.gif?raw=true)  |  ![](https://github.com/Loris697/DecisionTransformers/blob/main/output_folder/starting_rewards0.5_0.gif) |  ![](https://github.com/Loris697/DecisionTransformers/blob/main/output_folder/starting_rewards1.0_1.gif)


## Overview 📖

The "Decision Transformer: Reinforcement Learning via Sequence Modeling" paper introduces an innovative framework that reconceptualizes reinforcement learning (RL) as a sequence modeling problem, employing the architectural principles of the Transformer. The core idea of the Decision Transformer is to utilize a causally masked Transformer architecture to predict optimal actions based on a sequence of past states, actions, and desired future returns (rewards). Unlike traditional RL approaches that involve complex value function estimation or policy gradient methods, the Decision Transformer simplifies the process by conditioning an autoregressive model on these sequences. This enables the model to generate future actions that aim to achieve specified reward outcomes, effectively allowing the adjustment of an agent's actions based on the desired reward levels.The model demonstrates strong performance across various benchmarks, including Atari games and OpenAI Gym tasks, often matching or surpassing state-of-the-art model-free offline RL baselines. The approach bypasses the need for explicit reward discounting and value function approximation, instead utilizing the Transformer's ability to model long sequences for direct credit assignment, which is particularly effective in environments with sparse or misleading rewards. [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://proceedings.neurips.cc/paper_files/paper/2021/file/7f489f642a0ddb10272b5c31057f0663-Paper.pdf). This reimplementation is made mostly for fun but shoild also aims providing modular, easy-to-extend code.

## Repository Structure 📁

This repository contains several Jupyter notebooks that help in managing the workflow from data preparation to model training and evaluation:

- **`CreateTrajectory.ipynb`**: Generates trajectories using several pre-trained models located in `../models/{env_id}`. The trajectories are stored for later training.
- **`ReadRandomTrajectory.ipynb`**: Utility notebook for debugging that reads and displays a trajectory from the saved episodes using the `SequenceExtractor` class.
- **`Training.ipynb`**: Handles the training loop for the Decision Transformer, reading the saved trajectories and performing the learning process.
- **`TestModel.ipynb`**: Evaluates the performance of the trained models on their respective environments to test the effectiveness of the Decision Transformer.

## Getting Started 🚀

To get started with this repository, follow the steps below:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Loris697/DecisionTransformers/
   cd DecisionTransformers
   ```

2. **Install Dependencies**:
   Ensure you have Jupyter Notebook or JupyterLab installed, along with other necessary Python libraries. Install them using:
   
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Models**:
   Place your trained model files in the respective `../models/{env_id}` directories as expected by the `CreateTrajectory.ipynb` notebook.

5. **Run the Notebooks**:
   Open the notebooks in Jupyter and run them in sequence to perform trajectory creation, training, and testing.

## Requirements 🛠️

- Python 3.x
- Jupyter Notebook or JupyterLab
- Other dependencies listed in `requirements.txt` (not yet available)

## Contributing 🤝

We welcome contributions to this project. If you have suggestions for improvements or bug fixes, feel free to open an issue or create a pull request.

## License 📄

Specify the license under which your project is made available.

## Contact 📧

For any additional questions or comments, please reach out via GitHub issues or directly through email at [loriscino97@gmail.com](mailto:loriscino97@gmail.com).
