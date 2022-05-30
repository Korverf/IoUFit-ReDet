from torch.autograd import Function

from .. import os2_riroi_align_cuda
from mmdet.core import rroi2roi


class OS2RiRoIAlignFunction(Function):

    @staticmethod
    def forward(ctx, features, rois, out_size, spatial_scale, sample_num=0, nOrientation=8):
        if isinstance(out_size, int):
            out_h = out_size
            out_w = out_size
        elif isinstance(out_size, tuple):
            assert len(out_size) == 2
            assert isinstance(out_size[0], int)
            assert isinstance(out_size[1], int)
            out_h, out_w = out_size
        else:
            raise TypeError(
                '"out_size" must be an integer or tuple of integers')
        ctx.spatial_scale = spatial_scale
        ctx.sample_num = sample_num
        ctx.save_for_backward(rois)
        ctx.feature_size = features.size()
        ctx.nOrientation = nOrientation

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)     #rois: (index, x, y, w, h, angle)

        enclosed_rois = rroi2roi(rois)     #旋转proposal的外包水平框
        mask = features.new_zeros(batch_size, num_channels, data_height, data_width)
        obb_mask = rroi2mask(rois, mask)
        features = features * obb_mask
        output_cls = features.new_zeros(num_rois, num_channels, out_h, out_w)
        output_reg = features.new_zeros(num_rois, num_channels, out_h, out_w)
        if features.is_cuda:
            os2_riroi_align_cuda.forward(features, enclosed_rois, rois, out_h, out_w, spatial_scale,
                                     sample_num, nOrientation, output_cls, output_reg)
        else:
            raise NotImplementedError

        return output_cls, output_reg

    @staticmethod
    def backward(ctx, grad_output_cls, grad_output_reg):
        feature_size = ctx.feature_size
        spatial_scale = ctx.spatial_scale
        sample_num = ctx.sample_num
        nOrientation = ctx.nOrientation
        rois = ctx.saved_tensors[0]
        assert (feature_size is not None and grad_output_cls.is_cuda)
        assert (feature_size is not None and grad_output_reg.is_cuda)

        #grad_output = (grad_output_cls + grad_output_reg) / 2.
        #grad_output = grad_output_cls + grad_output_reg
        batch_size, num_channels, data_height, data_width = feature_size
        out_w = grad_output_cls.size(3)
        out_h = grad_output_cls.size(2)

        grad_input = grad_rois = None
        if ctx.needs_input_grad[0]:
            grad_input = rois.new_zeros(batch_size, num_channels, data_height,
                                        data_width)
            os2_riroi_align_cuda.backward(grad_output_cls.contiguous(), grad_output_reg.contiguous(), rois, out_h,
                                      out_w, spatial_scale, sample_num, nOrientation,
                                      grad_input)

        return grad_input, grad_rois, None, None, None, None


os2_riroi_align = OS2RiRoIAlignFunction.apply
