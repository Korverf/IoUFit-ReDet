#python tools/train.py configs/HRSC2016/OS_ReDet_re50_refpn_3x_hrsc2016_rerun.py
#
#python tools/test.py configs/HRSC2016/OS_ReDet_re50_refpn_3x_hrsc2016_rerun.py \
#    work_dirs/OS_ReDet_re50_refpn_3x_hrsc2016_rerun_2/epoch_36.pth \
#    --out work_dirs/OS_ReDet_re50_refpn_3x_hrsc2016_rerun_2/results.pkl
#
#python tools/parse_results.py --config configs/HRSC2016/OS_ReDet_re50_refpn_3x_hrsc2016_rerun.py --type OBB \
#--output_path ./work_dirs/OS_ReDet_re50_refpn_3x_hrsc2016_rerun_2
#
#python DOTA_devkit/hrsc2016_evaluation.py --det_root ./work_dirs/OS_ReDet_re50_refpn_3x_hrsc2016_rerun_2

python tools/train.py configs/HRSC2016/ReDet_re50_refpn_3x_hrsc2016_rerun.py

python tools/test.py configs/HRSC2016/ReDet_re50_refpn_3x_hrsc2016_rerun.py \
    work_dirs/ReDet_re50_refpn_3x_hrsc2016_rerun_4/epoch_36.pth \
    --out work_dirs/ReDet_re50_refpn_3x_hrsc2016_rerun_4/results.pkl

python tools/parse_results.py --config configs/HRSC2016/ReDet_re50_refpn_3x_hrsc2016_rerun.py --type OBB \
--output_path ./work_dirs/ReDet_re50_refpn_3x_hrsc2016_rerun_4

python DOTA_devkit/hrsc2016_evaluation.py --det_root ./work_dirs/ReDet_re50_refpn_3x_hrsc2016_rerun_4