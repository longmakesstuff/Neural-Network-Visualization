# Neuronal network in Kotlin

Neuronal network is a statistical model where parameters can be updated with back propagation on a target. The implementation 
in this repository supports following features:

- [x] Regression with MAE, MSE
- [ ] Classification with cross entropy
- [ ] Gradient descent optimization
- [ ] Momentum driven gradient descent
- [ ] Nestorov driven gradient descent
- [ ] Adagrad
- [ ] Adam

### Dependencies
- `Apache Commons CSV` for CSV reading.
- `XChart` for visualization.
- `Nd4j` for linear algebra.

### Initial performance evaluation

The performance validation of the neuronal network was made on the advanced [House Price Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).
For the training purpose, the data of the dataset has to be processed. Each discrete column will be one-hot-encoded and each continuous-valued column will be 
scaled to `[0, 1]`.

![](images/performance.png)