compile = True
max_iters = 10000
lr_decay_iters = 10000
wandb_log = False
eval_interval = 50
out_dir = "out"
config_of_data = "config/dataset_config.json"

batch_size = 900
block_size = 700  # Max size of observation
max_action_size = 20

# Model size 7M:
n_layer = 8
n_head = 8
n_embd = 256

# Model size 2M:
# n_layer = 5
# n_head = 5
# n_embd = 160

# Critic
num_bins = 400  # = num bins per action (20) * action size
max_value = 5
alpha_cql = 1
loss_type = "QT"
num_sampled_actions = 4
calculate_rtg = True
gamma = 0.95
group_reward = True
sigma_loss = 0.01
target_type = "TD"

# max value for pos encoding:
history_len = 10
n_attr = 129
n_agents = 20
n_team = 2

gradient_accumulation_steps = 16

# init_from = 'resume'
