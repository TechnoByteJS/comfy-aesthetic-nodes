# Aesthetic Nodes
Two aesthetic scoring models, one based on the same as A1111, the other based on Image Reward.

These run on CPU, so you don't need a (NVIDIA) GPU.

## Aesthetic Scorer

This node will output a predicted aesthetic score as a number and display it with the appropriate node (e.g., rgthree's ["Any"](https://github.com/rgthree/rgthree-comfy#display-any) node).   

You can load a number of scoring models, such as:

https://github.com/grexzen/SD-Chad/blob/main/chadscorer.pth

https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/ava+logos-l14-linearMSE.pth


## Image Reward

This node will output a predicted aesthetic score as a number and display it with the appropriate node (e.g., rgthree's ["Any"](https://github.com/rgthree/rgthree-comfy#display-any) node).

The difference between this node and the Aesthetics Scorer is that the underlying ImageReward is based on Reward Feedback Learning (ReFL) and uses 137K input samples that were scored by humans.  It often scores much lower than the Aesthetics Scorer, but not always!

## Credits

+ These nodes are based on [tusharbhutt/Endless-Nodes](https://github.com/tusharbhutt/Endless-Nodes)

