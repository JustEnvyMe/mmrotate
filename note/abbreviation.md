- `misc` means 

- `MS` means multiple scale image split.
- `RR` means random rotation.

- `with_cp` 在 backbone 中设置 `with_cp=True`。这使用 PyTorch 中的 `sublinear strategy` 来降低 backbone 占用的 GPU 显存。
- `grad_clip` gradient clipping
- `ms` in variable names means `multi-stage`
- `ss` simple scale
- `rr` random rotation
- with_cp (bool): Use checkpoint or not. Using checkpoint will save some
  memory while slowing down the training speed. Default: False.
- `GD` GDLoss: Gaussian Distribution
