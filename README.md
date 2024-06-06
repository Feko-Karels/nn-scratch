# nn-scratch

This is a basic neural network for MNIST classification which does not use any machine learning frameworks.
The network consist of three fully connected layers, softmax for the output layer and cross entropy loss:
<ul>
  <li>Input layer: 784</li>
  <li>Second layer: 80 + ReLU</li>
  <li>Output layer: 10 + Softmax</li>
</ul>

![nn](https://github.com/Feko-Karels/nn-scratch/assets/76912802/53fb6d46-f079-43aa-aba6-65cbf9eb2cc8)

With basic sdg and a reasonble training time it reaches about 91% accuracy on test data.
```
Epoch: 490, runing since 488.4 seconds
Average loss is:  1.8067085883157994
Training Set Accuracy is:  0.9113666666666667
Traing is done
Test Accuracy is: 0.91185
```
