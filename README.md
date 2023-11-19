# Dewey
Dewey is a fast reproducible training automation tool for MLOps pipelines.

Dewey is a machine learning automation tool written to create consistent reproducible ways to train models in a framework agnostic way. It allows providing a training specification, and the Dewey training framework takes care of all of the standard boilerplate code involving writing training loops, monitoring & metrics, managing model checkpoints, and more.

![](https://media.giphy.com/media/129g9HK07tEtZm/giphy.gif)

**Please note:** Don't expect much... YET! This repository is currently in alpha development. It is subject to rapid updates and breaking API changes. But please stay posted. Good stuff is on the way.

## Usage
Install using pip:

`pip install pydewey`

Create a file called `train.py`. Define your model, loss, and optimizer(s), and Dew(ey) it up real good!

For an example, put the `train.py` file in this repo's `examples` folder in your current directory, and run the command `dwy`.

## Todos:
* remove `next` input var from plugin module functions
* install plugin deps on use (torch, tf, etc.)
* plugin deps on other plugins
* tensorflow plugin
* keras plugin
* doc
* test
* optimize code
* distributed training
* better support multi-model workflows
