import mmcv
from mmcv import Config
from mmrotate.datasets import build_dataset

"""
直接生成pkl文件，没有带参数没有生成submission的zip
"""
def format(cfg_file, pkl_file, out_dir):
    cfg = Config.fromfile(cfg_file)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(pkl_file)
    dataset.format_results(outputs, submission_dir=out_dir)


if __name__ == '__main__':
    cfg = 'configs/roi_trans/roi_trans_swin_tiny_fpn_1x_dota_le90_1024.py'
    pkl = "work_dirs/roi_trans_swin_tiny_fpn_1x_dota_le90_1024/result.pkl"
    out_dir = "work_dirs/roi_trans_swin_tiny_fpn_1x_dota_le90_1024/result"
    format(cfg, pkl, out_dir)
