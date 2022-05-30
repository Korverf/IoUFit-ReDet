#python tools/parse_results.py --config configs/HRSC2016/IOUfit_ReDet_re50_refpn_3x_hrsc2016.py --type OBB \
#--output_path ./work_dirs/IOUfit_ReDet_re50_refpn_3x_hrsc2016_9

#python tools/parse_results.py --config configs/HRSC2016/ReDet_re50_refpn_3x_hrsc2016_rerun.py --type OBB \
#--output_path ./work_dirs/ReDet_re50_refpn_3x_hrsc2016_rerun_2

python tools/parse_results.py --config configs/IOUFit_DOTA/IOUfit_ReDet_re50_refpn_1x_dota1_trainval.py --type OBB \
--output_path ./work_dirs/IOUfit_ReDet_re50_refpn_1x_dota1_trainval_13

#python tools/parse_results.py --config configs/HRSC2016/ReDet_re50_refpn_3x_hrsc2016_gwd.py --type OBB \
#--output_path ./work_dirs/ReDet_re50_refpn_3x_hrsc2016_gwd
#
#python DOTA_devkit/hrsc2016_evaluation.py --det_root ./work_dirs/ReDet_re50_refpn_3x_hrsc2016_gwd