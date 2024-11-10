import numpy as np
import gym
from gym import spaces

class PortfolioEnv(gym.Env):
    """
    Custom Environment for Portfolio Optimization using Reinforcement Learning.
    This environment handles asset returns and portfolio weights, calculating
    reward based on Differential Sharpe Ratio.
    """
    def __init__(self, returns, dates, lookback_period=12, eta=1/12):
        super(PortfolioEnv, self).__init__()

        # Initialize environment parameters
        self.returns = returns
        self.dates = dates
        self.lookback_period = lookback_period
        self.num_assets = returns.shape[1]
        self.current_step = lookback_period
        self.eta = eta  # Sharpe Ratio adjustment factor

        # Define action space (portfolio weights) and observation space
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(self.num_assets, lookback_period + 1), dtype=np.float32)

        # Initialize portfolio and Sharpe Ratio variables
        self.portfolio_allocation = np.ones(self.num_assets) / self.num_assets  # Equal weights initially
        self.A_t = 0  # Sharpe ratio numerator
        self.B_t = 0  # Sharpe ratio denominator

    def reset(self):
        """Resets the environment for a new episode."""
        self.current_step = self.lookback_period
        self.A_t = 0
        self.B_t = 0
        self.portfolio_allocation = np.ones(self.num_assets) / self.num_assets
        return self._next_observation()

    def _next_observation(self):
        """Generates the next observation by combining lookback returns and portfolio allocation."""
        lookback_returns = self.returns[self.current_step - self.lookback_period:self.current_step]
        portfolio_allocation_column = self.portfolio_allocation.reshape(-1, 1)
        obs = np.hstack((lookback_returns.T, portfolio_allocation_column))

        # Validate observation shape
        assert obs.shape == (self.num_assets, self.lookback_period + 1), f"Unexpected observation shape: {obs.shape}"
        if np.any(np.isnan(obs)):
            raise ValueError("NaN values found in observations")

        return obs

    def step(self, action):
        """Executes a step in the environment using the provided action (portfolio weights)."""
        self.current_step += 1

        # Normalize the action to ensure portfolio weights sum to 1
        action = np.clip(action, 1e-6, 1)
        action_sum = np.sum(action)
        if action_sum == 0 or np.isclose(action_sum, 0):
            raise ValueError("Action sum is zero or too close to zero, cannot normalize weights.")
        self.portfolio_allocation = action / action_sum

        # Calculate portfolio return and update Sharpe Ratio variables
        portfolio_return = np.dot(self.portfolio_allocation, self.returns[self.current_step])
        delta_A_t = portfolio_return - self.A_t
        delta_B_t = portfolio_return**2 - self.B_t
        self.A_t += self.eta * delta_A_t
        self.B_t += self.eta * delta_B_t

        # Compute reward as Differential Sharpe Ratio
        denom = (self.B_t - self.A_t**2)**(3/2)
        reward = 0 if denom == 0 else (self.B_t * delta_A_t - 0.5 * self.A_t * delta_B_t) / denom
        reward = np.nan_to_num(reward)  # Handle NaNs if they occur

        # Get the next observation and check if the episode is done
        obs = self._next_observation()
        done = self.current_step >= len(self.returns) - 1

        return obs, reward, done, {}

    def render(self, mode='human'):
        """Renders the environment state (if needed)."""
        print(f"Step: {self.current_step}, Portfolio Allocation: {self.portfolio_allocation}")
