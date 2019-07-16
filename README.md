# `battlesim`: Modelling and animating simulated battles between units in Python.

The aim of this side project is to become familiar with Python classes and with primitive forms of animation and simulating environments. We map units onto a 2D plane and run simple simulations that involve them moving towards an enemy unit and attacking it. Rounds finish when one team has completely wiped out the other side, or we have reached the maximum number of timesteps.

![Image not found](images/quiver1.svg)

The code for the primary engine is found in `battlesim/`, and implementations/examples are found in the Jupyter notebooks. Animations should display properly in these notebooks.

## How to use: The Basics

Firstly, check the requirements for using this simulator, of which most come with the **Anaconda** distribution. In addition you will need the **ffmpeg** video conversion package to generate the simulations as animations.

Secondly, you will need to import the package as:

```python
>>> import battlesim as bsm
```

We recommend using `bsm` as a shorthand to reduce the amount of writing out you have to do. If you're using Jupyter notebook we also recommend:

```python
>>> import matplotlib.pyplot as plt
>>> plt.rcParams["animation.html"] = "html5"
>>> %matplotlib inline
```

The second line is important when you come to plotting the animations, as there are a number of issues with using it. All of the heavy lifting comes in the `bsm.Battle` object that provides a neat interface for all of the operations you would like to conduct:

```python
>>> import battlesim as bsm
>>> battle = bsm.Battle("datasets/starwars-clonewars.csv")
>>> battle
<battlesim.battle.Battle at 0x7f6bb4458c88>
```

You can see that we have specified a 'dataset' from which all of the unit roster can be drawn from; for specifics of how this file should
be oriented, see the documentation. We then need to specify units to create to form an army. For example, in this Star Wars example, we could specify a play-off between Clone troopers and B1 battledroids:

```python
>>> battle.create_army([("B1 battledroid", 70), ("Clone Trooper", 50)])
```

Here we call the `create_army` function, which internally creates an efficient `numpy` matrix, ready to perform the simulation. This is stored in the `battle.M_` object, a heterogenous `ndarray` element. From here, we might also want to specify the locations of our different blobs, as by default they will be sitting on top of each other at (0, 0).

```python
>>> battle.apply_position_gaussian([(0, 1), (10, 1)])
```

Here the first element of each tuple represents the mean of the gaussian distribution, and the second element refers to the variance (or spread). From here, all we need to do now is simulate this:

```python
>>> F = battle.simulate()
```

By default, the simulation function will make a record of important parameters at each step and then return these parameters as a `pandas.DataFrame` at the end in *long form*. In addition, because you want to see what's going on - we can animate the frames using one of the functions in `simplot.py` (example below using Jupyter notebook):

```python
>>> from IPython.display import HTML
>>> HTML(bsm.quiver_fight(F).to_jshtml())
```

![Image not found](simulations/sim1.gif)

Here `quiver_fight` treats each unit object as a quiver arrow in 2-d space (position and direction facing it's enemy). The targets should move towards each other and attempt to kill each other. Dead units are represented as crosses **'x'** on the map. 

![Image not found](images/quiver2.svg)

The rest is for you to explore, tweak and enjoy watching arrows move towards each other and kill each other.

## One step further: Repeated runs

If you're interested in seeing how each team fare over multiple runs (to eliminate random biases), then `bsm.Battle` objects once defined, contain a `simulate_k()` method, where `k` specifies the number of runs you wish to complete. Unlike `simulate()` by itself, it does not return a `pandas.DataFrame` of frames, but rather the number of units from each team left standing at each iteration.

```python
>>> runs = battle.simulate_k(k=40)
```

This is the beginning of creating an interface similar to Machine Learning, whereby the outcome can be a classification (team) or regression (number of units surviving) target, and the unit compositions, aspects of the engine etc., can be inputs.


## Engine specifics: Coming up with Units

Units are provided in a file named `unit-scores.csv`, which provides the Name, Allegiance, HP (hit points) and other characteristics about each of the units.Battles are created using a unit file in `battle.py`. Simulations are called using one of the functions found in `simulator.py`, and the results can then be plotted using one of the plotting functions in `simplot.py`, alternatively you can write code to plot it yourself.

| Name | Allegiance | HP | Damage | ... | Accuracy |
| ------------- | ---------- | ---- | ---- | --- | ---- |
| B1 battledroid | CIS | 20 | 15 | ... | 30 |
| Clone Trooper | Republic | 50 | 16 | ... | 60 |
| Clone Commando | Republic | 120 | 20 | ... | 97 |
| Droideka | CIS | 20 | 20 | ... | 40 |

The example above, is some basic stats about Star Wars units as an illustration. This work is very much still in the earliest stages and may be subject to large changes in the future.

***

Ensure that any use of this material is appropriately referenced and in compliance with the license.