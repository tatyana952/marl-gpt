from pydantic import BaseModel


class PogemaInferenceConfig(BaseModel):
    num_agents: int = 13
    num_previous_actions: int = 5
    cost2go_value_limit: int = 20
    agents_radius: int = 5
    cost2go_radius: int = 5
    device: str = "cuda"
    context_size: int = 256
    mask_actions_history: bool = False
    mask_goal: bool = False
    mask_cost2go: bool = False
    mask_greed_action: bool = False
    grid_step: int = 64
    save_cost2go: bool = False
    batch_size: int = 2048
    task_type_id: int = -1
    num_process: int = 8
    vector_observation: bool = True
