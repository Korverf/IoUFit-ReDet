python tools/test.py configs/HRSC2016/ReDet_re50_refpn_3x_hrsc2016_rerun.py \
    work_dirs/ReDet_re50_refpn_3x_hrsc2016_rerun/epoch_36.pth \
    --out work_dirs/ReDet_re50_refpn_3x_hrsc2016_rerun/results_new_idea.pkl

python tools/parse_results.py --config configs/HRSC2016/ReDet_re50_refpn_3x_hrsc2016_rerun.py \
--type OBB --output_path ./work_dirs/ReDet_re50_refpn_3x_hrsc2016_rerun

python DOTA_devkit/hrsc2016_evaluation.py --det_root ./work_dirs/ReDet_re50_refpn_3x_hrsc2016_rerun
