# Big Data & Deep Learning Exercises

This repository contains a series of exercises that guided me through fundamental concepts of machine learning and deep learning. Starting with basic regression, the projects progressively increase in complexity, covering data manipulation with NumPy, building shallow and deep neural networks with PyTorch, and refactoring code for scalability. The series culminates in a computer vision capstone project for fruit classification.

## Project Structure

Each numbered directory represents a self-contained exercise that builds upon the concepts of the previous one.

### 1. Regression Models
* **Files:** `1_regression/`
* **Description:** This section introduces machine learning with basic regression. It demonstrates how to fit polynomial and linear models to time-series data (`overhead_data.txt`, `returns_data.txt`) by defining a Mean Squared Error (MSE) loss function and using `scipy.optimize.minimize` to find the optimal model parameters.

### 2. NumPy for Data Processing
* **Files:** `2_numpy/`
* **Description:** This exercise focuses on essential data manipulation and preparation techniques using NumPy. The Fashion-MNIST dataset is loaded, inspected, and normalized. The data is then programmatically filtered and split into two categories: 'garments' and 'others'. The resulting processed arrays are saved as `fashion_mnist_garments.npz` for use in subsequent deep learning tasks.

### 3. Shallow Neural Network
* **Files:** `3_shallow_neural_network/`
* **Description:** An introduction to neural networks using PyTorch. This exercise demonstrates the fundamental components of a network by building a shallow model with a single hidden layer. It uses manually engineered features (symmetry and aspect ratio) from the garment images to perform classification, illustrating the concepts of weighted sums and non-linear activations (`sigmoid`).

### 4. Deep Neural Network (DNN)
* **Files:** `4_deep_neural_network/`
* **Description:** This exercise transitions from manual feature engineering to a deep learning approach. A multi-layer feed-forward neural network is built using `torch.nn.Sequential` to classify garments vs. non-garments. The model learns features directly from the flattened 28x28 pixel data, showcasing the power of deep networks for feature extraction. The training loop, loss calculation (`MSELoss`), and optimization (`Adam`) are implemented.

### 5. Refactoring for Scalability
* **Files:** `5_refactoring/`
* **Description:** This section focuses on software engineering best practices by refactoring the code from the previous exercise. The model is encapsulated within a `GarmentClassifier` class (inheriting from `nn.Module`), and the training logic is organized into a reusable `Trainer` class. This modular structure makes the code cleaner, more scalable, and easier to manage.

### 6-8. Advanced Topics
* **Files:** `6_overfitting/`, `7_training_cycle/`, `8_convolutional_neural_networks/`
* **Description:** These sections explore critical strategies for optimizing model performance and generalization. Key topics include:
    * **Mitigating Overfitting:** Implementation of L1/L2 regularization and Dropout layers to prevent the model from memorizing training data.
    * **Training Cycle:** Establishment of a robust training and validation loop, utilizing Early Stopping to halt training at the optimal point.
    * **Convolutional Neural Networks (CNNs):** Introduction to CNN architectures, utilizing convolutional and pooling layers to effectively process spatial hierarchies in image data.

### 9. Capstone Project: Fruit Classification
* **Files:** `9_capstone_project/`
* **Description:** The final capstone project applies the full suite of learned concepts to the **Fruits360 dataset**. A custom Convolutional Neural Network (CNN) is architected to classify images across 33 distinct fruit classes. By integrating advanced regularization techniques the model achieves a **98% accuracy** on the test data, demonstrating high precision and generalization capability.

## Key Concepts Covered
* **Modeling:** Linear & Polynomial Regression
* **Data Processing:** NumPy for array manipulation, data normalization, and filtering.
* **Deep Learning Framework:** PyTorch
* **Neural Network Architectures:** Shallow Networks, Deep Feed-Forward Networks (DNNs), Convolutional Neural Networks (CNNs).
* **Core DL Concepts:** Loss Functions (MSE), Optimizers (Adam), Activation Functions (ReLU, Sigmoid), Backpropagation, Batching, and Epochs.
* **Regularization:** L1/L2 penalties, Dropout, and Early Stopping.
* **Code Organization:** Refactoring PyTorch code into modular classes for models and trainers.
* **Computer Vision:** Image classification with the Fashion-MNIST and Fruits360 datasets.

## How to Use
1.  Clone the repository to your local machine:
    ```bash
    git clone [https://github.com/mkueverling/big-data.git](https://github.com/mkueverling/big-data.git)
    ```
2.  Navigate into the cloned directory:
    ```bash
    cd big-data
    ```
3.  Each numbered folder is a self-contained exercise. Navigate into the desired folder (e.g., `cd 1_regression`).
4.  Ensure you have the necessary Python libraries installed. You can install them using pip:
    ```bash
    pip install numpy pandas matplotlib scikit-learn torch torchvision
    ```
5.  Open and run the Jupyter Notebook (`.ipynb` file) within the directory to explore the code and its output.
