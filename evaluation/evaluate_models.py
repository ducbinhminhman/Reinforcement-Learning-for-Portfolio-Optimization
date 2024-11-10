import numpy as np

def evaluate_model(env, returns, model, initial_investment=1000000):
    """
    Evaluate a trained model on the given environment and returns.
    """
    obs = env.reset()
    cumulative_rewards, fund_values = [], [initial_investment]
    current_fund = initial_investment

    for _ in range(len(returns) - env.envs[0].lookback_period):
        action, _ = model.predict(obs)
        obs, rewards, done, _ = env.step(action)
        
        # Calculate portfolio return and update fund
        portfolio_return = np.dot(action, returns[env.envs[0].current_step])
        current_fund *= (1 + portfolio_return)
        
        cumulative_rewards.append(rewards[0])
        fund_values.append(current_fund)
        
        if done:
            break

    return np.array(cumulative_rewards), np.array(fund_values)
