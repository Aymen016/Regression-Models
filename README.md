# 🎯 **Regression Model**

![Untitled design (3)](https://github.com/user-attachments/assets/a801b18f-7272-48f6-925f-fa1c84c1d667)


## 📚 **Introduction**
This project explores multiple regression models (Linear, Quadratic, Cubic, and higher-degree polynomials) to predict. The goal is to implement and compare models of increasing complexity, analyzing how the Mean Squared Error (MSE) changes with each model.

The project is part of an assignment to demonstrate regression techniques and model evaluation in Python.

## 📂 **Project Structure**
- **trainRegression.csv**: Contains the training dataset with X and R values.
- **testRegression.csv**: Contains the testing dataset where we need to predict the outcome.
- **Main Code**: Python script that implements the regression models and visualizes results.

## 🚀 **Goals**
- Implement and compare **Linear**, **Quadratic**, **Cubic**, **Quartic (4th degree)**, **Quintic (5th degree)**, and **Sextic (6th degree)** regression models.
- Visualize the regression lines for each model and analyze their performance using **Mean Squared Error (MSE)**.

## 🛠️ **Tools and Libraries**
We used the following Python libraries in this project:
- **NumPy**: For numerical operations and matrix manipulations.
- **Matplotlib**: For data visualization and plotting graphs.
- **Pandas**: For data manipulation and CSV handling.

```bash
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

## 🚀 **Approach**

### **Model Implementation**:
- **Linear Regression**: Fit a simple line to the data using the normal equation.
- **Polynomial Regression**: Fit higher-degree polynomial models (quadratic, cubic, etc.) using the same approach.
- For each model, the equation is represented as a system of linear equations in matrix form. We solve for the model parameters using **matrix inversion**.

### **Predictions**:
- Using the model parameters, we calculate the predicted values (**y_hat**) for the test data.

### **Model Evaluation**:
- We evaluate the model performance using **Mean Squared Error (MSE)** to compare the different models.

## 📊 **Results and Observations**
The following models were evaluated:
- **Linear Model**: Shows a simple relationship between the input (**X**) and output (**R**).
- **Quadratic Model**: Introduces a curve and performs better than the linear model.
- **Cubic Model**: Fits the data more closely, showing improved MSE compared to lower-degree models.
- **Quartic & Quintic Models**: The MSE continues to improve, but higher-degree models may start to overfit.
- **Sextic Model**: While this model performs better in training, it risks overfitting due to the high complexity of the polynomial.

### **Key Insights**:
- The **Cubic, Quartic, and Quintic models** offer a good balance between predictive power and complexity.
- The **Sextic model** exhibits overfitting, as indicated by a slight rise in the MSE when compared to the 5th-degree model.

## 📊 **Visualizations**
- **Training Data Plot**: Scatter plot for the training data and regression line.
- **Test Data Plot**: Plot the predicted regression line against the actual test data.

### **Example of Linear Model Plot:**
![linear-model](https://github.com/user-attachments/assets/bed05356-7d60-4701-8ff5-8d3d1c7335dc)

## 📑 **Mean Squared Error (MSE) Comparison**

| Model            | MSE             |
|------------------|-----------------|
| Linear           | 0.3159          |
| Quadratic        | 0.3260          |
| Cubic            | 0.0515          |
| Quartic          | 0.0500          |
| Quintic          | 0.0442          |
| Sextic           | 0.0445          |

**Lower MSE** indicates better model performance.

## 🧑‍💻 **How to Run the Project**

### **Prerequisites**
Ensure you have **Python** installed along with the required libraries:

```bash
pip install numpy matplotlib pandas
```

### Running the Code
 1.Clone the repository:
```bash
git clone 
```

 2.Run the script:
```bash
python regression_models.py
```

This will train the models, compute MSE, and display the plots for training and testing data.

## 🚧 **Future Improvements**
- Explore more advanced models such as **Ridge** or **Lasso regression** to regularize the model and reduce overfitting.
- Implement **cross-validation** to better evaluate model performance.
- Experiment with other regression techniques like **Support Vector Machines (SVM)** for regression.

## 💬 **Conclusion**
Through this project, we implemented and compared multiple polynomial regression models. We showed that higher-degree polynomials can improve the model's ability to fit the data but may also lead to overfitting if the degree is too high. The **Cubic** and **Quartic models** seem to offer the best trade-off between complexity and accuracy.
