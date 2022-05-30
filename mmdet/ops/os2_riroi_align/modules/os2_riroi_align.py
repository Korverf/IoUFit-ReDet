from torch.nn.modules.module import Module

from ..functions.os2_riroi_align import OS2RiRoIAlignFunction


class OS2RiRoIAlign(Module):

    def __init__(self, out_size, spatial_scale, sample_num=0, nOrientation=8):
        super(OS2RiRoIAlign, self).__init__()

        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.nOrientation = int(nOrientation)

    def forward(self, features, rois):
        return OS2RiRoIAlignFunction.apply(features, rois, self.out_size,
                                        self.spatial_scale, self.sample_num, self.nOrientation)
