import torch.nn as nn
#from mmcv.cnn import (build_activation_layer)
import torch
#from mmdet.ops.box_iou_rotated import obb_overlaps


class IOUfitModule(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super(IOUfitModule, self).__init__()
        # self.fc1 = FC(in_channels=in_features, feedforward_channels=hidden_features, out_channels=hidden_features,
        #               act_cfg=dict(type='GELU'), add_residual=False)
        # self.fc2 = FC(in_channels=in_features, feedforward_channels=hidden_features, out_channels=hidden_features,
        #               act_cfg=dict(type='GELU'), add_residual=False)
        # self.fc3 = FC(in_channels=in_features, feedforward_channels=hidden_features, out_channels=hidden_features,
        #               act_cfg=dict(type='ReLU'), add_residual=False)
        # self.fc4 = FC(in_channels=in_features, feedforward_channels=hidden_features, out_channels=hidden_features,
        #               act_cfg=dict(type='ReLU'), add_residual=False)
        self.fc1 = FC(in_channels=in_features, feedforward_channels=512, out_channels=512,
                      act_func='ReLU', add_residual=False, with_act=True, with_bn=False)
        self.fc2 = FC(in_channels=512, feedforward_channels=512, out_channels=512,
                      act_func='ReLU', add_residual=False, with_act=True, with_bn=False)
        self.fc3 = FC(in_channels=512, feedforward_channels=512, out_channels=512,
                      act_func='ReLU', add_residual=False, with_act=True, with_bn=False)
        self.fc4 = FC(in_channels=512, feedforward_channels=512, out_channels=512,
                      act_func='ReLU', add_residual=False, with_act=True, with_bn=False)
        self.fc5 = FC(in_channels=512, feedforward_channels=512, out_channels=1,
                      add_residual=False, with_act=False, with_bn=False)

        # self.out_fc = Linear(hidden_features,1)
        #self.sigmoid = nn.Sigmoid()
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.loss_weight = 10.0
        #self.scale_factor = 1

    def forward(self, pred, gt):
        input = torch.cat([pred, gt], dim=-1)
        # out = self.fc1(input)
        # out = self.fc2(out)
        # out = self.fc3(out)
        # out = self.fc4(out)
        #out = out * self.scale_factor
        # out = self.fc5(out)
        #out = self.sigmoid(out)
        out = self.fc1(input)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        return out

    # def loss(self, rbboxes1, rbboxes2, iou_fit_value):
    #     IoU_targets = obb_overlaps(rbboxes1, rbboxes2.detach(), is_aligned=True).squeeze(
    #         1).clamp(min=1e-6, max=1)

    #     loss_fit = self.mse_loss(iou_fit_value, IoU_targets.detach()).sqrt()
    #     loss_fit = self.loss_weight * loss_fit
    #     # loss_fit = ((iou_fit_value - IoU_targets.detach()).square() + 1).log().sqrt()

    #     return loss_fit


class FC(nn.Module):
    def __init__(self,
                 in_channels,
                 feedforward_channels,
                 out_channels,
                 with_act=True,
                 with_bn=True,
                 act_func='ReLU',
                 add_residual=True):
        super(FC, self).__init__()
        # self.in_channels = in_channels
        # self.feedforward_channels = feedforward_channels
        # self.act_cfg = act_cfg
        self.add_residual = add_residual
        layers = nn.ModuleList()
        #self.relu = build_activation_layer(dict(type='ReLU'))
        layers.append(nn.Linear(in_channels, out_channels))
        if with_bn == True:
            layers.append(nn.BatchNorm1d(out_channels))
        if with_act == True:
            if act_func=='ReLU':
                self.activate = nn.ReLU()
                layers.append(self.activate)

        self.layers = nn.Sequential(*layers)

    def forward(self, x, residual=None):
        """Forward function for `FFN`."""
        out = self.layers(x)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        return residual + out


def get_model(in_features, hidden_features=None):

    model = IOUfitModule(in_features, hidden_features)  # 101
    return model