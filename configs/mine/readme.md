
## 需要调整通用的参数

- max_epochs

- save_bset 

```shell
evaluation = dict(save_best='auto', interval=1, metric='mAP')
```    

- checkpoint

```shell
checkpoint_config = dict(interval=12, create_symlink=False)

```

- tensorboard

```shell
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
```