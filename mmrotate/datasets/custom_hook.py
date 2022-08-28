from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class ImageMetaIterHook(Hook):
    # todo

    def _check_head(self, runner):
        """logger iterating image file name to find out data error conveniently.

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
        # model = runner.model
        # dataset = runner.data_loader.dataset
        # if dataset.CLASSES is None:
        #     runner.logger.warning(
        #         f'Please set `CLASSES` '
        #         f'in the {dataset.__class__.__name__} and'
        #         f'check if it is consistent with the `num_classes` '
        #         f'of head')
        # else:
        #     assert type(dataset.CLASSES) is not str, \
        #         (f'`CLASSES` in {dataset.__class__.__name__}'
        #          f'should be a tuple of str.'
        #          f'Add comma if number of classes is 1 as '
        #          f'CLASSES = ({dataset.CLASSES},)')
        #     for name, module in model.named_modules():
        #         if hasattr(module, 'num_classes') and not isinstance(
        #                 module, (RPNHead, VGG, FusedSemanticHead, GARPNHead)):
        #             assert module.num_classes == len(dataset.CLASSES), \
        #                 (f'The `num_classes` ({module.num_classes}) in '
        #                  f'{module.__class__.__name__} of '
        #                  f'{model.__class__.__name__} does not matches '
        #                  f'the length of `CLASSES` '
        #                  f'{len(dataset.CLASSES)}) in '
        #                  f'{dataset.__class__.__name__}')

    def before_train_iter(self, runner):
        """Check whether the training dataset is compatible with head.

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
        self._check_head(runner)

    def before_val_iter(self, runner):
        """Check whether the dataset in val epoch is compatible with head.

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
        self._check_head(runner)
