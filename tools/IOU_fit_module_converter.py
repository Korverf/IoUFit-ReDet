import torch
from  collections import OrderedDict

saved_pth = torch.load('../work_dirs/iou_fit_module_49/iter_1740000.pth')
saved_state_dict = saved_pth['state_dict']
new_state=OrderedDict()
for k, v in saved_state_dict.items():
    if k=='fc1.layers.0.weight':
        new_state.update({'bbox_head.IOUfit.fc1.layers.0.weight':v})
    elif k=='fc1.layers.0.bias':
        new_state.update({'bbox_head.IOUfit.fc1.layers.0.bias': v})
    elif k=='fc2.layers.0.weight':
        new_state.update({'bbox_head.IOUfit.fc2.layers.0.weight': v})
    elif k=='fc2.layers.0.bias':
        new_state.update({'bbox_head.IOUfit.fc2.layers.0.bias': v})
    elif k == 'fc3.layers.0.weight':
        new_state.update({'bbox_head.IOUfit.fc3.layers.0.weight': v})
    elif k == 'fc3.layers.0.bias':
        new_state.update({'bbox_head.IOUfit.fc3.layers.0.bias': v})
    elif k == 'fc4.layers.0.weight':
        new_state.update({'bbox_head.IOUfit.fc4.layers.0.weight': v})
    elif k == 'fc4.layers.0.bias':
        new_state.update({'bbox_head.IOUfit.fc4.layers.0.bias': v})
    elif k == 'fc5.layers.0.weight':
        new_state.update({'bbox_head.IOUfit.fc5.layers.0.weight': v})
    elif k == 'fc5.layers.0.bias':
        new_state.update({'bbox_head.IOUfit.fc5.layers.0.bias': v})

for k, v in saved_state_dict.items():
    if k=='fc1.layers.0.weight':
        new_state.update({'rbbox_head.IOUfit.fc1.layers.0.weight':v})
    elif k=='fc1.layers.0.bias':
        new_state.update({'rbbox_head.IOUfit.fc1.layers.0.bias': v})
    elif k=='fc2.layers.0.weight':
        new_state.update({'rbbox_head.IOUfit.fc2.layers.0.weight': v})
    elif k=='fc2.layers.0.bias':
        new_state.update({'rbbox_head.IOUfit.fc2.layers.0.bias': v})
    elif k == 'fc3.layers.0.weight':
        new_state.update({'rbbox_head.IOUfit.fc3.layers.0.weight': v})
    elif k == 'fc3.layers.0.bias':
        new_state.update({'rbbox_head.IOUfit.fc3.layers.0.bias': v})
    elif k == 'fc4.layers.0.weight':
        new_state.update({'rbbox_head.IOUfit.fc4.layers.0.weight': v})
    elif k == 'fc4.layers.0.bias':
        new_state.update({'rbbox_head.IOUfit.fc4.layers.0.bias': v})
    elif k == 'fc5.layers.0.weight':
        new_state.update({'rbbox_head.IOUfit.fc5.layers.0.weight': v})
    elif k == 'fc5.layers.0.bias':
        new_state.update({'rbbox_head.IOUfit.fc5.layers.0.bias': v})
    else:
        new_state.update({k: v})



torch.save(new_state, '../work_dirs/iou_fit_module_49/iou_fit_module_new.pth')
# for old pytorch version to load:
# torch.save(new_state, '../work_dirs/iou_fit_module_49/iou_fit_module_new.pth', _use_new_zipfile_serialization=False)







