import numpy as np
from stable_baselines3 import PPO, DDPG, TD3, SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.portfolio_env import PortfolioEnv

def train_models(train_returns, train_dates, valid_returns, valid_dates, model_type='PPO'):
    """
    Train a reinforcement learning model on portfolio data with early stopping.
    """
    env_train = DummyVecEnv([lambda: PortfolioEnv(train_returns, train_dates)])
    env_valid = DummyVecEnv([lambda: PortfolioEnv(valid_returns, valid_dates)])
    
    hyperparams = {
        'learning_rate': 3e-4,
        'batch_size': 256,
        'gamma': 0.99
    }
    
    # Model selection
    model_class = {'PPO': PPO, 'DDPG': DDPG, 'TD3': TD3, 'SAC': SAC, 'A2C': A2C}[model_type]
    model = model_class('MlpPolicy', env_train, verbose=1, **hyperparams)
    
    patience, best_valid_reward, patience_counter = 10, -np.inf, 0
    
    for epoch in range(50):
        model.learn(total_timesteps=1014, reset_num_timesteps=False)
        
        # Evaluate on validation set
        obs = env_valid.reset()
        valid_cumulative_reward = sum([env_valid.step(model.predict(obs)[0])[1][0] for _ in range(len(valid_returns) - env_valid.envs[0].lookback_period)])
        
        # Early stopping and saving best model
        if valid_cumulative_reward > best_valid_reward:
            best_valid_reward = valid_cumulative_reward
            patience_counter = 0
            model.save(f"best_model_{model_type}.zip")
            print(f"Best model for {model_type} saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered for {model_type}")
                break

    return model_class.load(f"best_model_{model_type}.zip"), env_train, env_valid
