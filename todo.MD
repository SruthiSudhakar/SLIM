#Todo 

##July 7th:
1. running sweeps action_transformer_lr 
cd /proj/vondrick3/sruthi/Appaji/SLIM/Ctrl-World/
conda activate ctrl-world
  for STEP in 5000 10000

bash scripts/eval_ckpt.sh model_ckpt/big_changeA_lr1e5/checkpoint-25000.pt 2 robocasa_atomic_all 36 random 0
bash scripts/eval_ckpt.sh model_ckpt/big_changeA_lr1e5/checkpoint-25000.pt 3 robocasa_gr00t_rollouts 36 random 0


3. see how the trained model is doing on failures vs successes. write a script to check MSE loss (and think of any other metrics that may be needed specific to seeing if task success/failure is faithful (like policy eval))

##Later:
1. Add a warm start to eval script so we can evaluate what happens if there was history frames vs not

