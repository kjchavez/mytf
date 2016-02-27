# MyTensorFlow

Just a collection of useful abstraction on top of the TensorFlow Python API
that make it easier to create complex models in fewer lines of code.

There are two guiding principles:

1. Everything is a subgraph, in the spirit of TensorFlow, exposing named inputs
   and outputs.
2. All configuration parameters are accessible/modifiable from the top level
   module.


## Litmus tests

* I should be able to create an LSTM like this:

```python
X, y = SequenceData(filename="shakespeare.txt")
model = LSTM(input=X, output_dim=128)
# Note at this point X and model.input refer to the same tensor.

loss = SomeLossType(predictions=model.output, labels=y)
trainer = Trainer(model=model, loss=loss, method='SGD', max_iterations=1e6)
trainer.run()
```

* Starting from pre-trained weights should be a breeze.
* 

## Goals
* Become well acquainted with the TensorFlow python API.
* Facilitate exploring methods presented in publications.