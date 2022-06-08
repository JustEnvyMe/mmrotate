
## dataset

### samples_per_gpu
    
旧的：

```shell

data = dict(
  samples_per_gpu,
  workers_per_gpu,
  train = dict(
  
  ),
  val = dict(
  
  ),
  test = dict(
  
  )
)
  
# 这样只有train_loader 能配置samples_per_gpu，val_loader&test_loader都不能读取到samples_per_gpu
# 但是workers_per_gpu可以公用

```

新的配置

```shell

data = dict(
  
  train = dict(
    samples_per_gpu，
    workers_per_gpu,
  ),
  val = dict(
    samples_per_gpu，
    workers_per_gpu,
  ),
  test = dict(
    samples_per_gpu，
    workers_per_gpu,
  )
)
  
# 各自配各自的

```


## symlink

latest.pth 是一个软链接

ntfs磁盘创建软链接可能有问题

禁止symlink
```shell
checkpoint_config = dict(create_symlink=False)
```