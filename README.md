<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/moonwatcher-ai/moonwatcher/assets/735435/4e17639c-e82b-4f93-b70b-47472865e365">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/moonwatcher-ai/moonwatcher/assets/735435/bcbfd1ba-a3c6-420d-8cef-afc617205f83">
  <img alt="Logo Moonwatcher" src="https://github.com/moonwatcher-ai/moonwatcher/assets/735435/bcbfd1ba-a3c6-420d-8cef-afc617205f83">
</picture>
<h1 align="center" weight='300' >The Evaluation & Testing framework for Computer vision models</h1>
<h3 align="center" weight='300' >Control performance risks, bias and security issues in AI models</h3>
<div align="center">

  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/moonwatcher-ai/moonwatcher/blob/main/LICENSE)
  [![Moonwatcher on Discord](https://img.shields.io/discord/1230407128842506251?label=Discord)](https://discord.com/invite/xHgSYGXZQK)

</div>

## Install Moonwatcher 🌝

```sh
pip install moonwatcher
```

## Try the demos
> [!WARNING]
> The demos require `wget` to be installed on your system.

In the demo the performance of a model on unusual values for brightness, contrast and saturation of the underlying
dataset are checked. To see how to create your own specific test scenarios check out [Quickstart](#quickstart).

Object detection (the demo will download the val2017 set of COCO and use a subset of it):
```sh
python -m moonwatcher.demo_detection
```

Classification (the demo will download STL-10 as a dataset):
```sh
python -m moonwatcher.demo_classification
```

# Contents

- 🏃‍♀️ **[Quickstart](#quickstart)**
    - **1**. 🧑‍🏫 [Slices, Checks and Checksuites](#slices_checks_and_checksuites)
      - 🍰 [Slices](#slices)
      - ✅ [Checks](#checks)
      - 📄 [Checksuites](#checksuites)
    - **2**. 🤖 [Run automated checks](#automated-checks)
    - **3**. 👨‍💻 [Write custom checks and checksuites](#write-custom-checks-and-checksuites)
- 🖥️ **[Web app](#webapp)**

<h1 id="quickstart">🏃‍♀️ Quickstart</h1>

<h2 id="slices_checks_and_checksuites">1. 🧑‍🏫 Slices, Checks and Checksuites</h2>
There are three core concepts (apart from models and datasets) to this framework. These concepts are called Checks, Checksuites and Slices.

<h3 id="slices"> Slices</h3>
A slice is a subset of a dataset. There are different methods in the framework to create those subsets for sophisticated evaluation and testing setups.

<h3 id="checks"> Checks</h3>
A check is defining one specific evaluation and/or testing setups. It defines the metric used, the dataset or slice to evaluate/test on and optionally the test comparison.  
When a check is applied on a specific model it returns the evaluation calculated and optionally the testing result (True/False).

<h3 id="checksuites"> Checksuites</h3>
A checksuite combines multiple checks into one. It is a suite of checks as the name suggests.



<h2 id="automated-checks">2. 🤖 Run automated checks</h2>  

Look into the relevant demo (demo_classification.py or demo_detection.py) to see how to create the MoonwatcherModel and MoonwatcherDataset from your data.
```python
from moonwatcher.check import automated_checking
from moonwatcher.model.model import MoonwatcherModel
from moonwatcher.dataset.dataset import MoonwatcherDataset

# Your model (your_model) and dataset (your_dataset) loading somewhere

# Look into the relevant demo (demo_classification.py or demo_detection.py)
# to see how to create the MoonwatcherModel and MoonwatcherDataset from your data.
mw_model = MoonwatcherModel(
  model=your_model,
  ...
)
mw_dataset = MoonwatcherDataset(
  dataset=your_dataset,
  ...
)

automated_checking(model=mw_model, dataset=mw_dataset)  
```

<h2 id="use-custom-checks-and-checksuites">3. 👨‍💻 Write custom checks and checksuites</h2>  

Writing a custom check works like this.
```python
from moonwatcher.check import Check

accuracy_check = Check(
    name="AccuracyCheck",
    dataset_or_slice=mw_dataset,
    metric="Accuracy",
    operator=">",
    value=0.8,
)

# and run it on your model:
check_result = accuracy_check(mw_model)
```
> [!TIP]
> You can also slice your dataset and use a slice for the check instead of the whole dataset.

> [!TIP]
> Class/category based checking is not yet supported, but will be part of the next iteration.

Now adding another check and combining both into a checksuite
```python
from moonwatcher.check import Check, CheckSuite

precision_check = Check(
    name="PrecisionCheck",
    dataset_or_slice=mw_dataset,
    metric="Precision",
    operator=">",
    value=0.8,
)

# Combine them into a checksuite
first_checksuite = CheckSuite(
    name="AllChecks", checks=[accuracy_check, precision_check]
)

# and run it on your model:
checksuite_result = first_checksuite(mw_model)
```

<h1 id="Webapp"> 🖥️ Web app</h1>
The package can be used on its own, 
is open-source and will always be. We additionally developed a web app you
can use to visualize results in a nice way. To try it out, check out 

[Web app instructions](readme/README_webapp.md).  
  
⭐️ Don’t forget to star the project if you want to support open source testing of ML models.

That's it. Have fun! 🌚
