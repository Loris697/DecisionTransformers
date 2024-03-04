from torch import Dataset

def collate_fn(data):
    """
       data: dict for the X and Y
    """
    Xs, Ys = zip(*data)
    
    rewards = torch.zeros((len(data),) +  Xs[0]["reward"].shape)
    observations = torch.zeros((len(data),) +  Xs[0]["observation"].shape)
    actions = torch.zeros((len(data),) +  Xs[0]["action"].shape)
    
    labels = torch.tensor(np.array(Ys))

    #print(labels.shape)
    #print(rewards.shape)
    #print(observations.shape)
    #print(actions.shape)

    for i in range(len(data)):
        rewards[i] = Xs[i]["reward"]
        observations[i] = Xs[i]["observation"]
        actions[i] = Xs[i]["action"]

    return {
             "rewards": rewards,
             "observations": observations,
             "actions":actions,
            }, labels

class SequenceExtractor(Dataset):
    def __init__(self, env, seq_len = 32, dataset_len = 16384):
        self.seq_len = seq_len
        self.dataset_len = dataset_len
        self.env = env
        self.env_name = env.unwrapped.envs[0].spec.id

    def __len__(self):
        return self.dataset_len

    ## il seed è usato come indice
    def __getitem__(self, seed):
        if seed is not None:
            random.seed(seed)
        # List all subfolders in the folder
        models_subfolder = [f.path for f in os.scandir(self.env_name) if f.is_dir()]
        
        # Choose a random subfolder
        random_model_subfolder = random.choice(models_subfolder)
        
        files = [f.path for f in os.scandir(random_model_subfolder) if f.is_file()]
        assert len(files) > 0, f"The number of file in the folder {random_model_subfolder} should be greater 0)"
        random_file = random.choice(files)

        # Read the Parquet file into a DataFrame
        df = pd.read_parquet(random_file)
        df['observation'] = df.apply(lambda row : row["observation"].reshape(self.env.observation_space.shape), axis = 1)

        #checking if the sequence is long enought (+1 to be sure to have Y too)
        assert len(df) > self.seq_len + 1, f"The lenght of the experience sequence ({x}) should be greater than sequence lenght + 1({y+1})"
        starting_row = random.randint(0, len(df) - (self.seq_len) -1)
        ending_row = starting_row + self.seq_len

        #converting to numpy array
        reward = torch.Tensor(np.stack(df["rewards"][starting_row:ending_row])).unsqueeze(1)
        observation = torch.Tensor(np.stack(df["observation"][starting_row:ending_row])).permute(0, 3, 1, 2)
        input_action = torch.Tensor(np.stack(df["action"][starting_row:ending_row]))

        #print(rewards.shape, observation.shape, input_action.shape)

        X = {
            "reward":reward, 
            "observation":observation, 
            "action":input_action,
            }
        Y = np.stack(df["action"][starting_row + 1:ending_row + 1])

        #print(Y.shape)

        return X, Y