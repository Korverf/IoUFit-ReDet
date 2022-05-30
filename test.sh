#python tools/test.py configs/HRSC2016/IOUfit_ReDet_re50_refpn_3x_hrsc2016.py \
#    work_dirs/IOUfit_ReDet_re50_refpn_3x_hrsc2016_9/epoch_36.pth \
#    --out work_dirs/IOUfit_ReDet_re50_refpn_3x_hrsc2016_9/results.pkl

python tools/test.py configs/HRSC2016/ReDet_re50_refpn_3x_hrsc2016_rerun.py \
    work_dirs/ReDet_re50_refpn_3x_hrsc2016_rerun/epoch_36.pth \
    --out work_dirs/ReDet_re50_refpn_3x_hrsc2016_rerun/results.pkl

#python tools/test.py configs/IOUFit_DOTA/IOUfit_ReDet_re50_refpn_1x_dota1_trainval.py \
#    work_dirs/IOUfit_ReDet_re50_refpn_1x_dota1_trainval_10/epoch_12.pth \
#    --out work_dirs/IOUfit_ReDet_re50_refpn_1x_dota1_trainval_10/results.pkl

#python tools/test.py configs/IOUFit_DOTA/ReDet_re50_refpn_1x_dota1_trainval_7.py \
#    work_dirs/ReDet_re50_refpn_1x_dota1_trainval_7/epoch_12.pth \
#    --out work_dirs/ReDet_re50_refpn_1x_dota1_trainval_7/results.pkl

#python tools/test.py configs/IOUFit_DOTA/ReDet_re50_refpn_1x_dota1_piou.py \
#    work_dirs/ReDet_re50_refpn_1x_dota1_piou_3/epoch_12.pth \
#    --out work_dirs/ReDet_re50_refpn_1x_dota1_piou_3/results.pkl