import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.portfolio_env import PortfolioEnv
from data.load_data import load_data
from models.train_models import train_models
from evaluation.evaluate_models import evaluate_model
from utils.seed import set_global_seed

# Set seed for reproducibility
set_global_seed(42)

if __name__ == "__main__":
    # Load data
    train_returns, valid_returns, test_returns, train_dates, valid_dates, test_dates = load_data("data/5_Industry_Portfolios.CSV")
    
    # Train model and get the best one
    best_model, train_env, valid_env = train_models(train_returns, train_dates, valid_returns, valid_dates, 'PPO')
    
    # Evaluate on the test set
    print("Evaluating on Test Set")
    test_env = DummyVecEnv([lambda: PortfolioEnv(test_returns, test_dates)])
    test_cumulative_rewards, test_fund_values = evaluate_model(test_env, test_returns, best_model)
    
    print("Training and evaluation completed successfully.")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(test_cumulative_rewards), label='Test Cumulative Rewards')
    plt.plot(test_fund_values, label='Test Fund Value')
    plt.legend()
    plt.title('Test Set Evaluation')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()
