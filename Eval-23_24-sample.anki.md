# 2. MTDE::1.2. Regression::Lab Evaluation 23/24 sample

<style>
hr { border-style: dashed; }
</style>

## <!-- MARK: Eval23-24 sample Q1 -->
Let `model_trained` be a regression model which has been previously fitted. What
is the appropiate way to compute the value of the coefficient $R^2$ over the test
set? Please note that `X_test_s` represents the (standardized) set of test
samples and `y_test` denotes the correct (or desired) values?

* [ ] a) `r2_score=model_trained.score(X_test_s,y_test)`
* [ ] b) `r2_score=model_trained.fit(X_test_s,y_test)`
* [ ] c) `r2_score=model_trained.predict(X_test_s,y_test)`
* [ ] d) `r2_score=model_trained.score(X_test_s)`

***

* [x] a) r2_score=model_trained.score(X_test_s,y_test)

## <!-- MARK: Eval23-24 sample Q2 -->
Indicate the appropiate way to divide a dataset (`X`, `y`) into 3 subsets: a
training subset (`X_train`, `y_train`) that contains 50 % of the total data, a
validation subset (`X_val`, `y_val`) that contains 25 % of the total data, and
finally, a test subset (`X_test`, `y_test`) including the remaining 25 % of the
total data.

* [ ] a)

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)
    X_val, X_test, y_val, y_test = train_test_split(X,y, test_size = 0.25)
    ```

* [ ] b)

    ```python
    X_aux, y_train, y_aux = train_test_split(X,y, test_size = 0.5)
    X_val, X_test, y_val, y_test = train_test_split(X_train,y_ rain, test_size = 0.5)
    ```

* [ ] c)

    ```python
    X_train, X_aux, y_train, y_aux = train_test_split(X,y, train_size = 0.5)
    X_val, X_test, y_val, y_test = train_test_split(X_aux,y_aux, test_size = 0.25)
    ```

* [ ] d)

    ```python
    X_train, X_aux, y_train, y_aux = train_test_split(X,y, train_size = 0.5)
    X_val, X_test, y_val, y_test = train_test_split(X_aux,y_aux, test_size = 0.5)
    ```

***

* [x] d)

    ```python
    X_train, X_aux, y_train, y_aux = train_test_split(X,y, train_size = 0.5)
    X_val, X_test, y_val, y_test = train_test_split(X_aux,y_aux, test_size = 0.5)
    ```

## <!-- MARK: Eval23-24 sample Q3 -->
Which of the following statements is **incorrect**?

* [ ] a) In a linear regression model, it is not necessary to carry out
  hyperparameter optimization.
* [ ] b) In a Ridge regression model, it is possible to optimize the value of
  the hyperparameter `alpha`.
* [ ] c) In a Lasso regression model, it is possible to optimize the value of
  the hyperparameter `alpha`.
* [ ] d) In a KernelRidge regression model, it is possible to carry out the
  optimization of both `alpha` and `beta` hyperparameters.

***

* [x] d) In a KernelRidge regression model, it is possible to carry out the
  optimization of both `alpha` and `beta` hyperparameters.

## <!-- MARK: Eval23-24 sample Q4 -->
When designing a KernelRidge regression model with a polinomical kernel, Student
A decides to optimize the values of the hyperparameters `alpha` and `degree`
according to the following ranges of values:

```python
alpha = [0.001, 0.01, 0.1, 1, 10, 100]
degree = [2, 4, 6, 8, 10, 12]
```

However, Student B decides to explore the following values:

```python
alpha = [0.001, 0.01, 0.1, 1, 10, 100, 200]
degree = [2, 4, 6, 8, 10]
```

Which student will fit more regression models in the validation stage?

* [ ] a) Both of them will fit the same number of models
* [ ] b) Student A
* [ ] c) Student B
* [ ] d) None of the above answers is correct

***

* [x] b) Student A

## <!-- MARK: Eval23-24 sample Q5 -->
What indicates that there is a perfect fit in a regression model?

* [ ] a) The value $R^2 > 0$, which corresponds to RMSE = 1.
* [ ] b) The value $R^2 = 0$, which corresponds to RMSE = 1.
* [ ] c) The value $R^2 < 1$, which corresponds to RMSE = 0.
* [ ] d) The value $R^2 = 1$, which corresponds to RMSE = 0.

***

* [x] d) The value $R^2 = 1$, which corresponds to RMSE = 0.

## <!-- MARK: Eval23-24 sample Q6 -->
There are five basic steps when implementing a regression model:

* (a) Check the results of the fitted model to know whether the model is
  satisfactory.
* (b) Eventually do appropriate transformations to data to work with.
* (c) Apply the model for predictions (over the test set) and calculate the
  $R^2$ coefficient.
* (d) Import the packages and classes that you need.
* (e) Create the regression model and fit it with existing data.

However, those steps are currently listed in the wrong order. What’s the correct order?

* [ ] A) e, d, b, a, and c.
* [ ] B) d, b, e, a and c.
* [ ] C) e, c, a, b and d.
* [ ] D) d, e, c, b and a.

***

* [x] B) d, b, e, a and c.

## <!-- MARK: Eval23-24 sample Q7 -->
To test linear relationship of the true (or correct) values and estimated
values, which of the following plots has been used in the assignment?

* [ ] a) Barchart
* [ ] b) Scatter plot
* [ ] c) Histogram
* [ ] d) None of the above answers is correct

***

* [x] b) Scatter plot

## <!-- MARK: Eval23-24 sample Q8 -->
It is created a linear regression model as follows:

```python
from sklearn.linear model import LinearRegression
lr = LinearRegression()
```

This model is fitted by using the (standardized) training subset:

```python
lr.fit(X train s, y train val)
```

If it is printed the value of the attribute `.coef` of the model (`lr.coef_`),
what numeric valued would be printed out?

* [ ] a) 10
* [ ] b) 1
* [ ] c) 9
* [ ] d) None of the above answers is correct

***

* [x] c) 9

## <!-- MARK: Eval23-24 sample Q9 -->
A linear regression model has been fitted and the values of its attribute .coef have
been shown running the following lines of code:

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train_s, y_train_val)
print(lr.coef_)
```

The result (displayed on the screen) is:

```text
[3.65 -11 56.9 -40 -88 20]
```

How would you select the 2 most relevant features aiming at obtaining a reduced dataset?

* [ ] a)

    ```python
    columns selected = np.array([2, 5])
    X train reduced = X train s [:,columns selected]
    ```

* [ ] b)

    ```python
    columns selected = np.array([3, 4])
    X train reduced = X train s [:,columns selected]
    ```

* [ ] c)

    ```python
    columns selected = np.array([2, 4])
    X train reduced = X train s [:,columns selected]
    ```

* [ ] d)

    ```python
    columns selected = np.array([3, 5])
    X train reduced = X train s [:,columns selected]
    ```

***

* [x] c)

    ```python
    columns selected = np.array([2, 4])
    X train reduced = X train s [:,columns selected]
    ```

## <!-- MARK: Eval23-24 sample Q10 -->
How would you obtain the number of samples of the training subset
(`[X_train_s, y_train_val]`) in the assignment?

* [ ] a) `n_samples = X_train_s.shape[0]`
* [ ] b) `n_samples = X_train_s.shape[1]`
* [ ] c) `n_samples = y_train_val.shape[1]`
* [ ] d) `n_samples = len(X_train_s.shape[0])`

***

* [x] a) `n_samples = X_train_s.shape[0]`
