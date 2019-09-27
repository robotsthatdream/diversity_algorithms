 # coding: utf-8

from diversity_algorithms.algorithms.novelty_search import NovArchive, NovES
# No reason to expose other functions - they can be accessed through the submodule if needed

from diversity_algorithms.algorithms.behavior_descriptors import maze_behavior_descriptor, bipedal_behavior_descriptor

bd_funcs = dict()

grid_features = dict()

# FastsimSimpleNavigation-v0 related parameters
bd_funcs["FastsimSimpleNavigation-v0"] = maze_behavior_descriptor
grid_features["FastsimSimpleNavigation-v0"] = {
    "min_x": [0,0],
    "max_x": [600, 600],
    "nb_bin": 50
    }


bd_funcs["BipedalWalker-v2"] = bipedal_behavior_descriptor
grid_features["BipedalWalker-v2"] = {
    "min_x": [-600,-600],
    "max_x": [600, 600],
    "nb_bin": 50
    }

__all__=["novelty_search", "stats", "utils", "behavior_descriptors"]
