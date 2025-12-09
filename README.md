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

### Train offline TD3-BC

We first train a TD3-BC agent purely from the expert dataset:

```bash
python src/scripts/train_offline_td3_bc.py --config config/td3_bc.yaml
```

### Visualize best agent
online
```bash
python src/scripts/visualize_agent.py
```
and offline: 
```bash
python src/scripts/visualize_agent.py --model_dir ./models/td3_bc_offline_step_300000
``