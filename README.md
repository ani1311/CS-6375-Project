## Reinforcement Learning

### Requirements:
The following python packages need to be installed to run the code:

- numpy
- pygame
- matplotlib
- open ai gym

all of them can be installed using
`pip3 install numpy pygame gym matplotlib`

### Code brakdown
The experiments were prototyped in /experiments directory using jupyter
notebook, and then the file versons were written in main_code

```
.
+-- experiments: Jupyter notebook of experiments
|   +-- MonteCarloCartpole.ipynb : Prototype for Monte Carlo cartpole       
|   +-- MonteCarloCartpoleModel.json : Prototype for Monte Carlo gridworld
|   +-- TD0_Grid.ipynb : Prototype for TD0 cartpole
|   +-- TD0_CartPole.ipynb : Prototype for TD0 gridworld
|   +-- TDN_Grid.ipynb : Prototype for TDN cartpole
|   +-- TDN_cartpole.ipynb : Prototype for TDN gridworld
+-- main_code
|   +-- main.py : Main controller code
|   +-- monte_carlo_gridworld.py : Code runner for Monte Carlo cartpole
|   +-- monte_carlo_cartpole.py : Code runner for Monte Carlo gridworld
|   +-- TD0_gridworld.py : Code runner for TD0 cartpole
|   +-- TD0_cartpole.py : Code runner for TD0 gridworld
|   +-- TDN_gridworld.py : Code runner for TDN cartpole
|   +-- TDN_cartpole.py : Code runner for TDN gridworld
|   +-- (Dir) logs : All logs and code runner outputs are stored
|   +-- (Dir) saved-models: Model pickle files are stored
|   +-- gridworld.py: Env for gridworld
|   +-- model_util.py: Util file
|   +-- log_util.py: Util file
|   +-- TD_util.py: Util file
|   +-- monte_carlo_util.py: Util file
```     

### How to run code

After installing all dependency, run 
    `python3 main.py`
from main_code directory
