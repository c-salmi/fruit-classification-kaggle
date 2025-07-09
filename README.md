# Fruit Classification Kaggle Challenge

To sharpen my computer vision engineering skills, I have taken up a small kaggle challenge. The fruit dataset consists of 101 classes of fruits. Each class has around 400 images for the training set, 50 for validation set and finally 50 for the test set.

## Methodology

### EDA

Before getting started with any machine learning methods, I first want to understand the data better to make informed decision about what technique to use and why.

My **exploratory data analysis** (EDA) can be found in the `notebooks/eda.ipynb` jupyter notebook.

Based on my finding, we already have a decently large dataset and the samples seem quite diverse per class. This makes me think augmentation is not immediately needed. But I will come back to it at a later stage.

### Baseline

Now that we have inspected the data, I will start with a simple architecture and no data augmentation to achieve a solid baseline. From the baseline we will inspect the loss curves to judge how we can improve.
