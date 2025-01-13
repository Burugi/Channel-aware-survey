### Hyper parameter tuning
1. batch size - 32 fixed
2. n_trials - 30 좀 많은 것 같기도하고 -> 20으로 줄이고 / 그만큼 에폭도 많이 줄이기 
3. Example. <python main.py --model tcn --dataset milano --optimize --n_trial 50>