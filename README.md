# PortfolioRLProject

This project uses reinforcement learning to optimize portfolio management strategies. It includes a custom Gym environment for financial portfolios and applies algorithms from `stable-baselines3` to train, evaluate, and visualize portfolio returns.

## Project Structure

```
PortfolioRLProject/
├── envs/
│   └── portfolio_env.py       # Contains the PortfolioEnv class definition
├── models/
│   └── train_models.py        # Code to train the reinforcement learning models (PPO, A2C, etc.)
├── evaluation/
│   └── evaluate_models.py     # Function to evaluate trained models
├── utils/
│   └── seed.py                # Utilities for setting the global seed
├── data/
│   └── load_data.py           # Data loading and preprocessing
├── notebooks/
│   └── exploration.ipynb     # Jupyter notebook for exploration and visualization
├── main.py                       # Entry point for running the entire pipeline
└── requirements.txt              # Dependencies for the project
```

## Getting Started

### Prerequisites

Ensure you have Python 3.7 or higher installed. This project also requires `stable-baselines3`, `gymnasium`, `shimmy`, and `matplotlib`.

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ducbinhminhman/Reinforcement-Learning-for-Portfolio-Optimization.git
   cd PortfolioRLProject
   ```

2. Create a virtual environment (recommended):
   ```bash
    conda create --name reinforcementLearning python=3.12 
    conda activate reinforcementLearning
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Setting Up Data

Place your portfolio data in the `data/` folder. By default, `main.py` references `5_Industry_Portfolios.CSV` as the data file.

### Running the Project

1. **Train the Model**:
   Run `main.py` to load data, train models, evaluate them on a test set, and visualize results.
   ```bash
   python main.py
   ```

2. **Evaluate**:
   After training, `main.py` evaluates the model on a test set and displays performance metrics.

## Usage

The project uses reinforcement learning models like PPO, DDPG, TD3, SAC, and A2C to optimize portfolio allocations based on historical returns data.

### Customizing Training Parameters

To adjust model hyperparameters, modify `train_models.py` in the `models/` folder. You can switch the model type by changing the model name in `main.py` or `train_models.py`.

### Visualizations

`main.py` generates plots for cumulative rewards and fund value over time. These visualizations can help analyze the model’s performance.

## Project Components

1. **`envs/portfolio_env.py`**: Defines the custom Gym environment for the portfolio, including actions (portfolio weights) and rewards (based on differential Sharpe Ratio).

2. **`models/train_models.py`**: Contains code to train the reinforcement learning model, implementing early stopping and saving the best model.

3. **`models/evaluate_models.py`**: Evaluates a trained model by calculating cumulative rewards and fund values.

4. **`utils/load_data.py`**: Loads and preprocesses portfolio returns data for training, validation, and testing.

5. **`utils/seed.py`**: Sets a global random seed for reproducibility.

6. **`main.py`**: Runs the entire workflow from data loading to model training, evaluation, and visualization.

## Results

1. **Cumulative Returns**:
   - **PPO** achieved the highest cumulative return at **1450.94%**, followed by **DDPG** at **1189.49%**.
   - **SAC**, **A2C**, and the naive **1/N** portfolio had the lowest returns.

2. **Annualized Returns & Sharpe Ratios**:
   - **PPO** led with an annualized return of **12.01%** and the highest Sharpe ratio at **0.6190**, offering the best risk-adjusted returns.
   - The **1/N** portfolio showed the lowest volatility at **14.49%**.

3. **Maximum Drawdown**:
   - **SAC** had the smallest maximum drawdown at **-41.61%**, while **EPO** had the largest at **-61.27%**.

#### Performance Metrics

| Metric                | EPO   | PPO       | A2C   | DDPG     | TD3   | SAC   | 1/N    |
|-----------------------|-------|-----------|-------|----------|-------|-------|--------|
| **Annual Return**     | 8.55% | **12.01%** | 8.36% | 11.16%   | 9.13% | 8.49% | 8.47%  |
| **Cumulative Return** | 626.47% | **1450.94%** | 596.58% | 1189.49% | 726.87% | 615.97% | 614.11% |
| **Volatility**        | 16.34% | 16.18%    | 15.35% | 16.27%   | 15.23% | 14.87% | **14.49%** |
| **Sharpe Ratio**      | 0.4010 | **0.6190** | 0.4146 | 0.5631   | 0.4683 | 0.4362 | 0.4334 |
| **Max Drawdown**      | -61.27% | -51.47%  | -50.06% | -52.46%  | -46.46% | **-41.61%** | -46.98% |

#### Monthly Return Distribution

| Metric           | EPO    | PPO       | A2C    | DDPG     | TD3   | SAC    |
|------------------|--------|-----------|--------|----------|-------|--------|
| **Mean**         | 0.80%  | **1.06%** | 0.77%  | 1.00%    | 0.83% | 0.77%  |
| **Median**       | 1.14%  | 0.97%     | 1.31%  | **1.48%** | 1.28% | 1.22%  |
| **Skewness**     | -0.4295| -0.0913   | -0.4469| -0.4037  | -0.3362| -0.4435 |
| **Kurtosis**     | **1.6648** | 1.3672 | 0.9043 | 1.1388   | 0.6485| 0.8456 |

The **PPO** and **DDPG** portfolios led in returns, with **PPO** providing the best risk-adjusted performance. **SAC** minimized drawdown, showing resilience. **EPO** displayed high kurtosis and skewness, indicating a higher chance of extreme negative returns.


## Troubleshooting

- **Module Not Found**: Ensure the Python path points to the correct project root, and modules are in the right directories.
- **Data Loading Issues**: Check that the data file is in the correct format and directory.
- **Gym Compatibility**: Install compatible versions as specified in `requirements.txt` to avoid conflicts.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to contribute, suggest improvements, or raise issues!
