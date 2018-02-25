# RNN Modules

### Overview
This is a collection of RNN classes built as PyTorch Modules.

The collection includes the following architectures:

* GRU
* DoubleGRU
* LSTM
* [Nested LSTM](https://arxiv.org/abs/1801.10308)
* Residual GRU
* [Residual LSTM](https://arxiv.org/abs/1701.03360)
* Residual Nested LSTM
* [Residual RNN](https://arxiv.org/abs/1611.01457)

Each has been evaluated on a simple memorization task. The goal was to predict the next binary digit in a random sequence given the k digits prior.

![RNN Prediction Task](./figs/rnn_prediction_task.png)
###### Where the x values are a individual binary digits in the sequence and y<sub>hat</sub> is the RNN's prediction of the next binary digit (for the input of x<sub>t</sub>, the prediction is of x<sub>t+1</sub>).

### Result Figures
![10 digit sequence](./figs/seq10_comparison.png)

![20 digit sequence](./figs/seq20_comparison.png)

![30 digit sequence](./figs/seq30_comparison.png)

![40 digit sequence](./figs/seq40_comparison.png)

![full 40 digit sequence](./figs/seq40_full_comparison.png)

![200 digit sequence](./figs/seq200_comparison.png)

![full 200 digit sequence](./figs/seq200_full_comparison.png)
