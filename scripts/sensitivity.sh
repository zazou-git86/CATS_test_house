python -m experiments.hyperparameters --n-runs 5 --data-name gfn --experiment temperature --stride 1 --out-size 100 --proj-size 50 --crop-ratio-min .5 --crop-ratio-max 1
python -m experiments.hyperparameters --n-runs 5 --data-name gfn --experiment gamma --stride 1 --out-size 100 --proj-size 50 --crop-ratio-min .5 --crop-ratio-max 1
python -m experiments.hyperparameters --n-runs 5 --data-name gfn --experiment margin --stride 1 --out-size 100 --proj-size 50 --crop-ratio-min .5 --crop-ratio-max 1
python -m experiments.hyperparameters --n-runs 5 --data-name gfn --experiment embed_size --stride 1 --out-size 100 --proj-size 50 --crop-ratio-min .5 --crop-ratio-max 1
python -m experiments.hyperparameters --n-runs 5 --data-name gfn --experiment proj_size --stride 1 --out-size 100 --proj-size 50 --crop-ratio-min .5 --crop-ratio-max 1
python -m experiments.hyperparameters --n-runs 5 --data-name gfn --experiment batch_size --stride 1 --out-size 100 --proj-size 50 --crop-ratio-min .5 --crop-ratio-max 1


