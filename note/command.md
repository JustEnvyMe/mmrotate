## samba share directory

- dataset

```shell
mount -t cifs //10.231.6.120/dataset /work/workspace/wzw/dataset/DOTA -o username=alex,password=zya123456,uid=1000,gid=1000
```

- code

```shell
mount -t cifs //10.231.6.120/code /work/workspace/wzw/code/ -o username=alex,password=zya123456,uid=1000,gid=1000
```

## docker build

一定选择空的目录build，不然docker会把目录下所有文件都加载。（直观的就是docker build执行卡住不输出任何东西，过了很久build context to Docker daemon 44.94GB）

- 压缩mmrotate
- 新建一个空目录，把压缩文件放到这个目录里

```shell


```


## docker run

```shell
# 设置shm
docker run -d -it --name mmrotate --shm-size="32g" -v /work/workspace/wzw/code/mmrotate/:/mmrotate -v /work/workspace/wzw/dataset/:/dataset mmrotate:v0.3.0 bash
```

## train

```shell
python tools/train.py <configfile>
```

## test

- 如果是要提交服务器 `--formate-only`
  ```shell
    --formate-only 不执行eval，只有预测，然后打包dota提交格式
    --eval-options eval方式带的参数，如果是--formate-only，参数只有submission_dir，dota提交文件存放位置
   ```
  
  **submission_dir 不能是已存在的目录**
  ```shell
  python ./tools/dist_test.sh  \
    configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py \
    checkpoints/SOME_CHECKPOINT.pth 1 --format-only \
    --eval-options submission_dir=work_dirs/Task1_results
  ```

- 如果是单纯evaluate

  dota本身test dataset是没有annfile的，直接evaluate是会报错的。

  需要本地离线验证，就只能讲valid dataset当做测试集使用

    ```shell
    # test_dataset路径改为val的路径 
    python ./tools/dist_test.sh  \
      configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py \
      checkpoints/SOME_CHECKPOINT.pth --eval mAP
    ```
