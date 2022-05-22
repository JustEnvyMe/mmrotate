
# install 

## conda
```shell
conda create -n mmrotate python=3.8 -y
conda activate mmrotate

```

## torch
```shell
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit==11.0 -c pytorch # 6.120
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```


## mmrotate

### option
```shell
pip install openmim
mim install mmrotate
```

### mmcv-full
```shell
pip install --no-cache-dir mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html
pip install --no-cache-dir mmcv-full==1.4.5
```

### mmdet
```shell
pip install mmdet   
```

### mmrotate
```shell
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

## test 

```shell
    python demo/image_demo.py \
        demo/demo.jpg \
        work_dirs/oriented_rcnn_r50_fpn_1x_dota_v3/oriented_rcnn_r50_fpn_1x_dota_v3.py \
        work_dirs/oriented_rcnn_r50_fpn_1x_dota_v3/epoch_12.pth \
        demo/vis.jpg
```


# problem

## “utils/spconv/spconv/geometry.h”: No such file or directory

```shell
sudo apt install libboost-filesystem-dev
sudo apt install libboost-dev
```

## 编译失败

降版本，pip install 指定版本