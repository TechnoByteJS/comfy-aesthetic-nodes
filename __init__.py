from .aesthetic_nodes import *

NODE_CLASS_MAPPINGS = {
	"Aesthetic Scoring": AestheticNode_Scoring,	
	"Image Reward": AestheticNode_ImageReward,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"Aesthetic Scoring": "Aesthetic Scoring",
	"Image Reward": "Image Reward"
 }

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
 
