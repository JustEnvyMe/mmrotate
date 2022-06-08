

## OMP_NUM_THREADS & MKL_NUM_THREADS

```text
/mmrotate/mmrotate/utils/setup_env.py:39: UserWarning: Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
f'Setting OMP_NUM_THREADS environment variable for each process '
/mmrotate/mmrotate/utils/setup_env.py:49: UserWarning: Setting MKL_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
f'Setting MKL_NUM_THREADS environment variable for each process '
```

test多线程会检测这2个环境变量，如果config和env都没有给就会报这个warning

环境默认是没有设置这个的

