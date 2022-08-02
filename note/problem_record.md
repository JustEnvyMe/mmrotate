## OMP_NUM_THREADS & MKL_NUM_THREADS

```text
/mmrotate/mmrotate/utils/setup_env.py:39: UserWarning: Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
f'Setting OMP_NUM_THREADS environment variable for each process '
/mmrotate/mmrotate/utils/setup_env.py:49: UserWarning: Setting MKL_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
f'Setting MKL_NUM_THREADS environment variable for each process '
```

test多线程会检测这2个环境变量，如果config和env都没有给就会报这个warning

环境默认是没有设置这个的

## 实现RotatedDETR，移植DETR出了一系列错误

- require positional arguments 'in_channels'
- cannot assign module before Module.__init__() call
- init中的类属性没有初始化
- ...

涉及到的都是super(type, self).__init__()

1. 初始化方法缺参数

在DETR中是继承AnchorFreeHead,同时在DETR中调用的是super(AnchorFreeHead),AnchorFreeHead的父类方法没有初始化必填参数

RotatedDETRHead 继承的是RotatedAnchorFreeHead，如果super(继承的是RotatedAnchorFreeHead)，RotatedAnchorFreeHead继承的是AnchorFreeHead，AnchorFreeHead的init是有参数的，这样就会造成少参数

**解决**：

    1. 传所需的参数 
    2. 直接掉ANchorFreeHead的super，调到基类(没有参数)

使用1的方式，看到其他调RotatedDETRHead的类是这样做的。2的方法没验证是否正确，毕竟2个初始化方法有差别

2. init中的类属性没有初始化

跟父类有关系，因为RotatedDETRHead代码里直接调了RotatedDETR的其他方法

RotatedDETR的方法里调了类属性，但是类属性实在RotatedDETR的init里，还没有初始化

**解决**

调整super放的位置


3. cannot assign module before Module.__init__() call

torch的异常，torch要求必须是Module类

但是，由于super放到init最后，这时候该类还是一般Class类，而不是Module


### 最麻烦的问题

上述几个bug相互关联，需要调整init逻辑

**解决**

重新看了下RotatedAnchorFree的代码与AnchorFreeHead是一样的。 XD

直接继承AnchorFreeHead, OK。
