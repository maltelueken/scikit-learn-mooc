# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---


# %% [markdown]
# # First look at our dataset
#
# Assuming you are familiar with pandas dataframes, in this notebook we look at the
# necessary steps required before any machine learning takes place. It involves:

# * highlighting that it is crucial to inspect the data before building a model
# * looking at the variables in the dataset, in particular, differentiate
#   between numerical and categorical variables, which need different
#   preprocessing in most machine learning workflows;
# * visualizing the distribution of the variables to gain some insights into the
#   dataset.

# %% [markdown]
# ```{note}
# **To the instructor**
#
# Instead of live-coding this notebook, explain the important concepts before ML and show the descriptive statistics.
# TBC.
# ```


# %% [markdown]
# ## Loading the adult census dataset
#
# We use pandas to load the data.


# %%
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")


# %%
adult_census.head()


# %% [markdown]
# **Rows and columns**
#
# * Each row in the dataframe represents a "sample". In the field of machine learning or
# descriptive statistics, commonly used equivalent terms are "record",
# "instance", or "observation".
# * Each column represents a type of information that has been collected and is
# called a "feature". In the field of machine learning and descriptive
# statistics, commonly used equivalent terms are "variable", "attribute", or
# "covariate".
#

# %%
target_column = "class"
adult_census[target_column].unique()


# %% [markdown]
# **Target and input variables**
#
# * Our target variable is the column **class**.
# * It has two classes: `<=50K` (low-revenue) and
# `>50K` (high-revenue).
# * Thus, the prediction problem is a binary classification problem.
# * The columns other than class are input variables for our model.


# %% [markdown]
# **Categorical and numerical columns**
#
# * There are numerical and categorical columns in the data
# * Numerical columns take continuous values. Example: `"age"`
# * Categorical columns take a finite number of values. Example: "`native-country`"


# %%
numerical_columns = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
categorical_columns = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
all_columns = numerical_columns + categorical_columns + [target_column]

adult_census = adult_census[all_columns]

# %% [markdown]
# We can check the number of samples and the number of columns available in the
# dataset:

# %%
print(
    f"The dataset contains {adult_census.shape[0]} samples and "
    f"{adult_census.shape[1]} columns"
)

# %% [markdown]
# We can compute the number of features by counting the number of columns and
# subtract 1, since one of the columns is the target.

# %%
print(f"The dataset contains {adult_census.shape[1] - 1} features.")

# %% [markdown]
# ## Inspecting the data: individual columns
#
# Before building a predictive model, it is a good idea to look at the data:
#
# * maybe the task you are trying to achieve can be solved without machine
#   learning;
# * you need to check that the information you need for your task is actually
#   present in the dataset;
# * inspecting the data is a good way to find peculiarities. These can arise
#   during data collection (for example, malfunctioning sensor or missing
#   values), or from the way the data is processed afterwards (for example
#   capped values).


# %% [markdown]
#
# ### Visually inspecting numerical columns
#
# Let's look at the distribution of individual features, to get some insights
# about the data. We can start by plotting histograms, note that this only works
# for features containing numerical values:

# %%
_ = adult_census.hist(figsize=(20, 14))

# %% [markdown]
#
# We can already make a few comments about some of the variables:
#
# * `"age"`: there are not that many points for `age > 70`. The dataset
#   description does indicate that retired people have been filtered out
#   (`hours-per-week > 0`);
# * `"education-num"`: peak at 10 and 13. These are the number of years of education.
# * `"hours-per-week"` peaks at 40, this was very likely the standard number of
#   working hours at the time of the data collection;
# * most values of `"capital-gain"` and `"capital-loss"` are close to zero.

# %% [markdown]
# ### Inspecting categorical columns
#
# For categorical variables, we can look at the distribution of values:
# * We should do this for both the target variable and the input variables

# %%
adult_census[target_column].value_counts()


# %% [markdown]
#
# #### Class imbalance
# **Imbalance in the target variable**
#
# * Classes are slightly imbalanced, meaning there are more samples of one
# or more classes compared to others.
# * In this case, we have many more samples with `" <=50K"` than with `" >50K"`.
# * Class imbalance in the target variable happens often in practice
# and may need special techniques when building a predictive model.
# * For example in a medical setting, if we are trying to predict whether subjects
# may develop a rare disease, there would be a lot more healthy subjects than
# ill subjects in the dataset.
#


# %% [markdown]
#
# **Imbalance in the input data**
#

# %%
adult_census["sex"].value_counts()

# %% [markdown]
#
# * The data collection process led to an important imbalance
# between the number of male/female samples. Thus, our data are **not representative** of the US population.
# * Training a model with such data imbalance can cause
# disproportioned prediction errors for the under-represented groups.
# * This is a typical cause of
# [fairness](https://docs.microsoft.com/en-us/azure/machine-learning/concept-fairness-ml#what-is-machine-learning-fairness)
# problems if used naively when deploying a machine learning based system in a
# real life setting.
#


# %% [markdown]
# ## Inspecting relationships between columns
#

# %% [markdown]
# ### In pandas
#
# * We can use the crosstabulation feature from pandas to see how two variables are related

# %%
pd.crosstab(
    index=adult_census["education"], columns=adult_census["education-num"]
)

# %% [markdown]
#
# **Redundante columns**
# * For every entry in `"education"`, there is only one single corresponding
# value in `"education-num"`.
# * This shows that `"education"` and `"education-num"` give you the same information.
# * Thus, we can remove `"education-num"` without losing information.
# * Note that having redundant (or highly correlated) columns can be a problem for machine
# learning algorithms.
#
# We drop this column now and in all future notebooks.
#

# %%
adult_census = adult_census.drop(
    columns=["education"]
)  # duplicated in categorical column


# %% [markdown]
# #### Using pairplots
#
# Another way to inspect the relationship between variables is to do a `pairplot` and show how each
# variable differs according to our target, i.e. `"class"`. Plots along the
# diagonal show the distribution of individual variables for each `"class"`. The
# plots on the off-diagonal can reveal interesting interactions between
# variables.

# %%
import seaborn as sns

# We plot a subset of the data to keep the plot readable and make the plotting
# faster
n_samples_to_plot = 5000
columns = ["age", "education-num", "hours-per-week"]
_ = sns.pairplot(
    data=adult_census[:n_samples_to_plot],
    vars=columns,
    hue=target_column,
    plot_kws={"alpha": 0.2},
    height=3,
    diag_kind="hist",
    diag_kws={"bins": 30},
)

# %% [markdown]
# ## Towards making predictions: creating decision rules by hand
#
# By looking at the previous plots, we could create some hand-written rules that
# predict whether someone has a high- or low-income. For instance, we could
# focus on the combination of the `"hours-per-week"` and `"age"` features.

# %%
_ = sns.scatterplot(
    x="age",
    y="hours-per-week",
    data=adult_census[:n_samples_to_plot],
    hue=target_column,
    alpha=0.5,
)

# %% [markdown]
#
# * The data points (circles) show the distribution of `"hours-per-week"` and
# `"age"` in the dataset.
# * Blue points mean low-income and orange points mean high-income.
# * This part of the plot is the same as the bottom-left plot in the pairplot above.
#
# In this plot, we can try to find regions that mainly contains a single class
# such that we can easily decide what class one should predict. We could come up
# with hand-written rules as shown in this plot:

# %%
import matplotlib.pyplot as plt

ax = sns.scatterplot(
    x="age",
    y="hours-per-week",
    data=adult_census[:n_samples_to_plot],
    hue=target_column,
    alpha=0.5,
)

age_limit = 27
plt.axvline(x=age_limit, ymin=0, ymax=1, color="black", linestyle="--")

hours_per_week_limit = 40
plt.axhline(
    y=hours_per_week_limit, xmin=0.18, xmax=1, color="black", linestyle="--"
)

plt.annotate("<=50K", (17, 25), rotation=90, fontsize=35)
plt.annotate("<=50K", (35, 20), fontsize=35)
_ = plt.annotate("???", (45, 60), fontsize=35)

# %% [markdown]
# * In the region `age < 27` (left region) the prediction is low-income. Indeed,
#   there are many blue points and we cannot see any orange points.
# * In the region `age > 27 AND hours-per-week < 40` (bottom-right region), the
#   prediction is low-income. Indeed, there are many blue points and only a few
#   orange points.
# * In the region `age > 27 AND hours-per-week > 40` (top-right region), we see
#   a mix of blue points and orange points. It seems complicated to choose which
#   class we should predict in this region.
#
# **Some machine learning models work similarly to what we just did**
#
# * We choose the two thresholds (27 years and 40 hours) arbitrarily
# * Decision trees work similarly, but more systematically and in a more scalable way
#   * Decision trees are systematic: They choose the "best" splits based on data without human intervention or inspection.
#   * Decision trees scale: They easily handle data with many more columns, which is difficult for humans.
# * We cover decision trees later in this course.


# %% [markdown]
# ## Notebook Recap
#
# In this notebook we:
#
# * looked at the different kind of variables to differentiate between
#   categorical and numerical variables;
# * inspected the data with `pandas` and `seaborn`. Data inspection can allow
#   you to decide whether using machine learning is appropriate for your data
#   and to highlight potential peculiarities in your data.
#
# We made important observations (which will be discussed later in more detail):
#
# * if your target variable is imbalanced (e.g., you have more samples from one
#   target category than another), you may need special techniques for training
#   and evaluating your machine learning model;
# * having redundant (or highly correlated) columns can be a problem for some
#   machine learning algorithms;
# * contrary to decision tree, linear models can only capture linear
#   interactions, so be aware of non-linear relationships in your data.


# %% [markdown]
# ### Sources
#
# **Data**
#
# We use data from the 1994 US census that we downloaded from
# [OpenML](http://openml.org/).
#
# You can look at the OpenML webpage to learn more about this dataset:
# <http://www.openml.org/d/1590>
#
# **Fairness in ML**
#
# We recommend our readers to refer to [fairlearn.org](https://fairlearn.org)
# for resources on how to quantify and potentially mitigate fairness issues
# related to the deployment of automated decision making systems that rely on
# machine learning components.
#
