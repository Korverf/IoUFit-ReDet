from torch.nn.modules.module import Module

from ..functions.os_riroi_align import OSRiRoIAlignFunction


class OSRiRoIAlign(Module):

    def __init__(self, out_size, spatial_scale, sample_num=0, nOrientation=8):
        super(OSRiRoIAlign, self).__init__()

        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.nOrientation = int(nOrientation)

    def forward(self, features, rois):
        return OSRiRoIAlignFunction.apply(features, rois, self.out_size,
                                        self.spatial_scale, self.sample_num, self.nOrientation)
