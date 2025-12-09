# rl-point-maze
Reinforcement Learning in the Point Maze env using Gymnasium

## Build Project
Inside some conda env:
```bash
pip install -e .
```

## Run Scripts
### Generate expert dataset
```bash
python src/scripts/generate_expert.py
```

### Train online TD3
```bash
python src/scripts/train_online.py
```

### Visualize best agent
```bash
python src/scripts/visualize_agent.py
```
