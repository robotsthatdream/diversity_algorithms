 # coding: utf-8

from diversity_algorithms.algorithms.novelty_search import NovArchive, NovES
# No reason to expose other functions - they can be accessed through the submodule if needed

from diversity_algorithms.algorithms.behavior_descriptors import maze_behavior_descriptor, bipedal_behavior_descriptor

bd_funcs = dict()

bd_funcs["FastsimSimpleNavigation-v0"] = maze_behavior_descriptor
bd_funcs["BipedalWalker-v2"] = bipedal_behavior_descriptor

__all__=["novelty_search", "stats", "utils", "behavior_descriptors"]
