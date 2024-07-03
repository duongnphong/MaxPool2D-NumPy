# MaxPool2D-NumPy


### Output Sizing Calculation
```py
output_height = (input_height - kernel_height + 2 * padding) / stride + 1
output_width = (input_width - kernel_width + 2 * padding) / stride + 1
```