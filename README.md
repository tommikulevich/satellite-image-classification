# 🛰️ Satellite Image Classification

> ☣ **Warning:** This project was created during my studies for educational purposes only. It may contain non-optimal solutions.

### 📑 About

A neural network model has been developed for **classifying satellite images** using a specific set of data and assigning each image to one of several defined categories. The implementation has potential applications in facilitating the monitoring of environmental changes, supporting spatial planning, or in the process of land cover analysis.

> The application is written in **Python 3.10.12**, using PyTorch 2.0.1 and CUDA 11.7, in VS Code 1.88.1.

### 🗂️ Dataset

[EuroSAT](https://zenodo.org/record/7711810#.ZAm3k-zMKEA) dataset was used, consisting of **27.000 satellite images** sorted into **10 categories** (`AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`). The set was divided in a 70/15/15 ratio into training, validation, and test data respectively. The dimensions of the images (**64x64x3**) remained unchanged during the data preparation process. However, to enhance the model's ability to generalize, a series of transformations such as random horizontal and vertical flips, random rotations up to 45 degrees, and random color adjustments (brightness, contrast, saturation) were dynamically applied with each data load during training.

### 🧠 NN Architecture

**Residual Network** was implemented. Initially, data passes through a convolutional block (`ConvBlock`), consisting of a 3x3 convolutional layer, batch normalization, and ReLU activation. Subsequently, the data proceed to four sets of residual blocks (`ResBlock`), where the first set may include dimension reduction at the beginning of each set. Residual blocks are divided into two types: one type includes an additional convolutional block in the residual connection to ensure proper dimension matching, and the other type uses the ReLU function when adaptation is not needed. The final block (`FinalConv`) processes the data through adaptive average pooling and a linear layer, adjusting the dimensions to fit the anticipated number of output classes.

<p align="center">
  <img src="_readme-img/1-arch_part1.png?raw=true" alt="Main architecture">
</p>

<p align="center">
  <img src="_readme-img/2-arch_part2.png?raw=true" alt="Components of the architecture">
</p>

### 📈 Results 

Adam optimizer was used with a learning rate of $0.0001$ and a weight decay component set at $10^{-5}$. The loss function employed during training was cross entropy loss. The entire process spanned $80$ epochs, with a batch size of $128$ samples. 

Below is a plot of the training and validation loss.

<p align="center">
  <img src="_readme-img/3-plots_loss.png?raw=true" width=600 alt="Loss plot">
</p>

To evaluate network performance, metrics such as **precision**, **recall**, and **F1-score** were used. Precision provided information on the proportion of correctly classified images within the assigned category, while recall measured the model's ability to identify all images belonging to a particular class. The F1 score, a harmonic mean of precision and recall, served as an indicator of balanced classification accuracy assessment.

The table below shows the percentage values of these performance metrics for the training, validation, and test sets.

|Metric|Training Set (*last epoch*)|Validation Set (*last epoch*)|Test Set|
|---------|:-----:|:-----:|:-----:|
|Precision [%]|97.11|95.23|95.20|
|Recall [%]|97.08|94.78|94.83|
|F1-score [%]|97.09|94.94|94.96|

Below, randomly selected samples from the test set (one from each category) are presented along with the model’s prediction. In the bottom left corner, an example is shown where the model confused a pasture with a river.

<p align="center">
  <img src="_readme-img/4-results.png?raw=true" alt="Examples">
</p>
