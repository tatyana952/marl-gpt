# For smacv2
import numpy as np
import json
from envs.smac_env.config import SMAC2ArgsConfig
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
from smacv2.env import StarCraft2Env

attribute_to_indx: dict[str, int] = {
    "move_action_north": 0,
    "move_action_south": 1,
    "move_action_east": 2,
    "move_action_west": 3,
    "shootable": 4,
    "visible": 5,
    "distance": 6,
    "relative_x": 7,
    "relative_y": 8,
    "pos_x": 9,
    "pos_y": 10,
    "health": 11,
    "shield": 12,
    "bit_0": 13,
    "bit_1": 14,
    "bit_2": 15,
}

all_race_dict = {"m": [1, 0, 0], "s": [1, 0, 0], "z": [0, 1, 0], "h": [0, 0, 1]}


def get_value_from_str(s: str, A: dict[str, int]) -> int:
    """
    Extracts key from string s based on the known patterns and returns A[key].
    Raises KeyError if no matching key is found.
    """
    if s in A:
        return A[s]

    for key in A:
        pattern = f"_{key}_"
        if pattern in s:
            return A[key]

    for key in A:
        pattern = f"_{key}"
        if s.endswith(pattern):
            return A[key]

    raise KeyError(f"No key from dict 'attribute_to_indx' found in string '{s}'")


def parse_map_race(map_name):
    ally_units = set()
    enemy_units = set()

    # Split into ally and enemy parts
    if "_vs_" in map_name:
        ally_part, enemy_part = map_name.split("_vs_")
    else:
        # Assume symmetric match-up
        ally_part = enemy_part = map_name

    # Extract unit types (letters) for each side
    for part in ally_part:
        if part.isalpha():
            ally_units.add(part)

    for part in enemy_part:
        if part.isalpha():
            enemy_units.add(part)

    return {"ally": sorted(ally_units), "enemy": sorted(enemy_units)}


def generate_features(map_name, feature_dict):
    obs = {"ally": [], "enemy": []}
    pos = {"ally": [], "enemy": []}
    # type of agent
    race_dict = parse_map_race(map_name)
    for key in obs.keys():
        race_list = race_dict[key]
        if len(race_list) == 2:
            obs[key].append(0)
            pos[key].append(15)
        elif len(race_list) == 1:
            obs[key].extend(all_race_dict[race_list[0]])
            pos[key].extend([13, 14, 15])

    obs["own"] = obs["ally"].copy()
    pos["own"] = pos["ally"].copy()

    if not (9 in feature_dict["own"]):
        obs["own"].extend([0, 0])
        pos["own"].extend([9, 10])
    return obs, pos


def concatinate_feat_arrays(original_feat, add_feat):
    original_feat = np.array(original_feat)
    feat_array = np.array(add_feat)
    feat_array = np.tile(feat_array, (original_feat.shape[0], 1))
    return np.concatenate((original_feat, feat_array), axis=1)


class SC2Decomposer:
    def __init__(
        self,
        map_name: str,
        smac_v: int = 2,
        zero_padding: bool = False,
        with_own_coord: bool = False,
    ):
        if smac_v == 2:
            self.cfg = SMAC2ArgsConfig.get_scenario_config(map_name)
            self.env = StarCraftCapabilityEnvWrapper(**self.cfg.model_dump())
            self.n_agents = self.cfg.capability_config.n_units
            self.n_enemies = self.cfg.capability_config.n_enemies
        elif smac_v == 1:
            self.env = StarCraft2Env(map_name=map_name, obs_own_pos=with_own_coord)
            self.n_agents = self.env.map_params["n_agents"]
            self.n_enemies = self.env.map_params["n_enemies"]

        self.move_feats, _, _, self.own_feats, self.obs_nf_en, self.obs_nf_al = (
            self.get_obs_size()
        )
        self.unit_type_bits = self.env.map_params["unit_type_bits"]
        self.dict_attribute = dict()

        env_info = self.env.get_env_info()

        self._zero_padding = zero_padding and smac_v == 1
        if self._zero_padding:
            feat_dict = self.get_obs_original_attribute_as_dict()
            self._additional_obs, self._additional_pos = generate_features(map_name, feat_dict)

    def get_obs_size(self):
        nf_al = self.env.get_obs_ally_feats_size()[1]
        nf_en = self.env.get_obs_enemy_feats_size()[1]
        own_feats = self.env.get_obs_own_feats_size()
        move_feats = self.env.get_obs_move_feats_size()

        enemy_feats = self.n_enemies * nf_en
        ally_feats = (self.n_agents - 1) * nf_al

        return move_feats, enemy_feats, ally_feats, own_feats, nf_en, nf_al

    def decompose_obs(self, obs_input):
        """
        obs_input: env_obs + last_action + agent_id
        env_obs = [move_feats, enemy_feats, ally_feats, own_feats]
        """

        # extract move feats
        move_feats = obs_input[: self.move_feats]
        # extract enemy_feats
        base = self.move_feats
        enemy_feats = [
            obs_input[base + i * self.obs_nf_en : base + (i + 1) * self.obs_nf_en]
            for i in range(self.n_enemies)
        ]
        # extract ally_feats
        base += self.obs_nf_en * self.n_enemies
        ally_feats = [
            obs_input[base + i * self.obs_nf_al : base + (i + 1) * self.obs_nf_al]
            for i in range(self.n_agents - 1)
        ]
        # extract own feats
        base += self.obs_nf_al * (self.n_agents - 1)
        own_feats = obs_input[base : base + self.own_feats]

        # own
        # own_obs = th.cat([move_feats, own_feats], dim=0)
        own_obs = np.concatenate([move_feats, own_feats])

        if self._zero_padding:
            own_obs, enemy_feats, ally_feats = self.obs_with_zero_padding(
                own_obs, enemy_feats, ally_feats
            )

        return own_obs, enemy_feats, ally_feats

    def obs_with_zero_padding(self, own_obs, enemy_feats, ally_feats):
        new_own_obs = np.concatenate(
            [own_obs, np.array(self._additional_obs["own"])]
        )
        new_enemy_feats = concatinate_feat_arrays(
            enemy_feats, self._additional_obs["enemy"]
        )
        new_ally_feats = concatinate_feat_arrays(
            ally_feats, self._additional_obs["ally"]
        )
        return new_own_obs, new_enemy_feats, new_ally_feats

    def add_feat(self, feat_dict):
        if self._zero_padding:
            feat_dict["ally"].extend(self._additional_pos["ally"])
            feat_dict["own"].extend(self._additional_pos["own"])
            feat_dict["enemy"].extend(self._additional_pos["enemy"])

    def get_obs_shape(self):
        obs_en = self.obs_nf_en
        obs_al = self.obs_nf_al
        obs_own = self.move_feats + self.own_feats

        if self._zero_padding:
            obs_en += len(self._additional_pos["enemy"])
            obs_al += len(self._additional_pos["ally"])
            obs_own += len(self._additional_pos["own"])

        return {
            "enemy": (self.n_enemies, obs_en),
            "ally": (self.n_agents - 1, obs_al),
            "own": (obs_own,),
        }

    def get_obs_original_attribute_as_dict(self) -> dict[str, list[int]]:
        obs_str = self.env.get_obs_feature_names()
        res = dict()

        # move
        # print(f"move: {obs_str[:self.move_feats]}")
        own_list = list()
        for attr in obs_str[: self.move_feats]:
            own_list.append(get_value_from_str(attr, attribute_to_indx))

        # own
        # print(f"own: {obs_str[-self.own_feats:]}")
        for attr in obs_str[-self.own_feats :]:
            own_list.append(get_value_from_str(attr, attribute_to_indx))

        # enemy
        enemy_list = list()
        # print(f"enemy: {obs_str[self.move_feats:self.move_feats+self.obs_nf_en]}")
        for attr in obs_str[self.move_feats : self.move_feats + self.obs_nf_en]:
            enemy_list.append(get_value_from_str(attr, attribute_to_indx))

        # ally
        ally_list = list()
        # print(f"enemy: {obs_str[-self.obs_nf_al-self.own_feats:-self.own_feats]}")

        for attr in obs_str[-self.obs_nf_al - self.own_feats : -self.own_feats]:
            ally_list.append(get_value_from_str(attr, attribute_to_indx))

        return {
            "enemy": enemy_list,
            "ally": ally_list,
            "own": own_list,
        }
         
    def get_obs_attribute_as_dict(self) -> dict[str, list[int]]:
        res_dict = self.get_obs_original_attribute_as_dict()
        self.add_feat(res_dict)
        return res_dict

    def get_obs_attribute_as_str(self) -> list[int]:
        obs_str = self.env.get_obs_feature_names()
        res = list()
        for attr in obs_str:
            res.append(get_value_from_str(attr, attribute_to_indx))
        return res


def get_all_features():
    # from envs.smac_env.decomposer import *
    map_list = ["protoss", "terran", "zerg"]
    dict_attr = dict()
    for map_ in map_list:
        map_name = f"{map_}_5_vs_5"
        decompose = SC2Decomposer(map_name)
        dict_attr[map_] = decompose.get_obs_attribute_as_dict()
    with open("attr_for_all_map.json", "w") as json_file:
        json.dump(dict_attr, json_file)

def t():
    from envs.smac_env.config import SMAC2ArgsConfig, SMACGPTInferenceConfig
    from envs.smac_env.smac_v2 import SMACv2, LogSMACWrapper
    from envs.smac_env.wrapper import InferenceSmacWrapper, SeparatedObs
    from envs.smac_env.decomposer import SC2Decomposer
    cfg = {"map_name": "3m", "obs_own_pos": False}
    env = SMACv2(cfg, 1)
    decomposer = SC2Decomposer("3m", 1, True, False)
    res = env.reset()
    o = res[0][0]['obs']
    decomposer.decompose_obs(o)


if __name__ == "__main__":
    get_all_features()
