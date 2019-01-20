# Convolutional Autoencoders (PyTorch)

An interface to setup Convolutional Autoencoders. It was designed specifically for model selection, to configure architecture programmatically. 

The configuration using supported layers (see ConvAE.modules) is minimal. Adding new type of layers is a bit painful, but once you understand what create_layer() does, all that's needed is to update ConvAE.modules and corresponding book-keeping in create_layer().

I/o dimensions for each layer are computed automatically. If the network has repeated blocks, they can be added without modifying class (or adding new code) by simply increasing depth.

## Example
```python
feature_maps = 32
depth = 10
pooling_freq = 1e100 # large number to disable pooling layers
strided_conv_freq = 2
strided_conv_feature_maps = 32
code_size = 8
input_dim = (1,64,64)

CONV_ENC_BLOCK = [("conv1", feature_maps), ("relu1", None)]
CONV_ENC_LAYERS = ConvAE.create_network(CONV_ENC_BLOCK, depth, 
                                    pooling_freq=pooling_freq,
                                    strided_conv_freq=strided_conv_freq, 
                                    strided_conv_channels=strided_conv_feature_maps)
CONV_ENC_NW = CONV_ENC_LAYERS + [("flatten1", None), ("linear1", 2 * code_size), ("linear1", code_size)]
model = ConvAE.ConvAE(input_dim, enc_config=CONV_ENC_NW)
```

## Auto-generated Encoder
```python
print(model.encoder)

ModuleList(
  (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (1): ReLU()
  (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (3): ReLU()
  (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  (5): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (6): ReLU()
  (7): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (8): ReLU()
  (9): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  (10): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (11): ReLU()
  (12): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (13): ReLU()
  (14): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  (15): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (16): ReLU()
  (17): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (18): ReLU()
  (19): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  (20): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (21): ReLU()
  (22): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (23): ReLU()
  (24): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  (25): Reshape()
  (26): Linear(in_features=128, out_features=16, bias=True)
  (27): Linear(in_features=16, out_features=8, bias=True)
)
```

## Auto-generated Decoder
```python
print(model.dncoder)
ModuleList(
  (0): Linear(in_features=8, out_features=16, bias=True)
  (1): Linear(in_features=16, out_features=128, bias=True)
  (2): Reshape()
  (3): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  (4): ReLU()
  (5): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (6): ReLU()
  (7): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (8): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  (9): ReLU()
  (10): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (11): ReLU()
  (12): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (13): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  (14): ReLU()
  (15): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (16): ReLU()
  (17): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (18): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  (19): ReLU()
  (20): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (21): ReLU()
  (22): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (23): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  (24): ReLU()
  (25): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (26): ReLU()
  (27): ConvTranspose2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
)
```
