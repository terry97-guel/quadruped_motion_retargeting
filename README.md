# Requirements
## Python versions
#### Recommanded python version: 3.11.9 (Compatible to mujoco-mpc repo) 
#### Tested python version: 3.11.9, 3.8.13 
*Some mujoco functions & plotting functions are not compatible to some python3 versions. It's generally okay, but if you have trouble plotting, try another python versions, mujoco-python-viewer, or mujoco.  (Currently, requirement.txt is set as mujoco-python-viewer== 0.1.4, mujoco==3.1.6).

## Summary of Subpackages
[mjtools](https://github.com/terry97-guel/mjtools): Useful mujoco functions \
[robot_menagerie](https://github.com/terry97-guel/robot_menagerie): XML files commonly used through out.\
[spatial_motion_retargeting](https://github.com/terry97-guel/spatial_motion_retargeting): Retargets motion for floating-base system at kinematic level.\
[MANN-Menagerie](https://github.com/terry97-guel/MANN-menagerie): Dataset introduced in paper [Mode-adaptive neural networks for quadruped motion control
](https://dl.acm.org/doi/10.1145/3197517.3201366).

# Installation 
There are two different installation methods, depending on your purpose.

- If you want to quickly start using this code, everything can be installed with two lines of code from [here](#for-casual-developer). However, you would not be able to change code for the subpackages, but only the source code of the repo.

- If you are devoted developer who wants to get hands on every subpackages, follow this [guide](#for-devoted-developer).

## For Casual developer
Comming soon

## For devoted developer
Initalize git submodules
```
git submodule update --init
```

Install each subpackages manually.
The order of installation matters.
```
cd robot_menagerie
pip install -e .
cd ..
```

```
cd mjtools
pip install -e .
cd ..
```

```
cd MANN-menagerie
pip install -e .
cd ..
```

```
cd spatial_motion_retargeting
pip install -e .
cd ..
```

```
cd quadruped_motion_mjpc/python
python setup_new.py install
cd ../..
```
Beware: Unlike other subpackages, [quadruped_motion_mjpc](./quadruped_motion_mjpc/) involves compiling c++ code. Therefore, if you change code in c++ side, make sure to **re-install** with `python setup_new.py install`.

# Main function
- kinematic_mr.py: Use spatial motion retargeting to transfer motion at kinematic level
- dynamic_mr.py (TODO): Use mjpc to transfer motion at dynamic level