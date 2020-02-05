# PyTorch - Basics

Need to read:

1. Broadcasting from Numpy
2. The first link in the Pytorch Roadmap
    - get all the functions and understand the important ones (access, reduction and element-wise)
3. Factory functions in OOPs

## Basic details about Tensors:

```py
data.dtype
data.device
data.cuda()
data.layout

data.shape
len(data.shape) #rank of the tensor
data.numel()
data.nelement()
```

## Ways to create Pytorch tensor tensor:
### with existing data:

```py
data = np.array([1,2,3])
type(data)

# The odd ball - converts the data to floating point float32
# class constructor
torch.Tensor(data)

# rest of the examples below create the tensors of the dtype as data

# factory function
torch.tensor(data)

torch.as_tensor(data)
torch.as_numpy(data)

```
Constructor uses `torch.get_default_dtype()` to figure the output tensor's dtype but the factory methods use type inference to figure the output tensor's type. Factory functions allow us to send the dtypes explicitly while passing the data. For example, `torch.tensor(data, dtype=torch.float64)`

The below 2 functions share the data values, which means if we modify `data` then `t3`, `t4` also will change. 
```py
t3 = torch.as_tensor(data)
t4 = torch.as_numpy(data)
```
But the other 2 functions create a copy of the data and use that. so modifying `data` wouldn't chnage anything else.

Best practice is to use `torch.tensor()` always and only when needed for performance boost use `as_tensor()`

### with no data:

```py
torch.eye(2)
torch.zeros(4,2)
torch.ones(3,1)
torch.rand(2,4)
```

## Tensor Operation:

1. Reshaping
```py
t.reshape()
t.view()
t.squeeze()
t.unsqueeze()

data.reshape()
data.reshape(1,-1)
data.view()

data.squeeze()
data.unsqueeze(dim=0)

image_batch.shape = (batch_size, n_color_channels, n_rows, n_columns)

flatten = lambda t:t.reshape(1,-1).squeeze()
# how to achieve flatten with only reshape function --> RESEARCH THIS!!!
# Answer -->
t.reshape(1,-1)[0]
t.reshape(-1)
t.view(-1)
t.flatten() 
# <-- end answer

#start_dim argument which axis to start flattening the tensor from. To skip batch_size from image_batch above, use 
image_batch.flatten(start_dim=1) === image_batch.reshape(3, -1)
image_batch.flatten(start_dim=2) === image_batch.reshape(3, 1, -1)



torch.stack([t1, t2, t3]) # all 3 need to be of the same shape
torch.concat([t1, t2, t3]) # all 3 need to be of the same shape along the concat axis


data.prod()
# all operations need to happen only between tensors within the same device and of the same dtype
```
2. Element-wise
    - Concept of broadcasting is introduced here
    - Read about broadcasting from Numpy website and write about it
    - Also all the element-wise
3. Access methods
    - Masked tensors, etc.
4. Reduction methods
    - Understand how dim works within these types of functions and implement them in a few rank-3 tensors to see the inner workings.