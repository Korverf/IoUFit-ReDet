CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/HRSC2016/IOUfit_ReDet_re50_refpn_3x_hrsc2016.py

CUDA_VISIBLE_DEVICES=1 python tools/test.py configs/HRSC2016/IOUfit_ReDet_re50_refpn_3x_hrsc2016.py \
    work_dirs/IOUfit_ReDet_re50_refpn_3x_hrsc2016_9_rerun/epoch_36.pth \
    --out work_dirs/IOUfit_ReDet_re50_refpn_3x_hrsc2016_9_rerun/results.pkl

CUDA_VISIBLE_DEVICES=1 python tools/parse_results.py --config configs/HRSC2016/IOUfit_ReDet_re50_refpn_3x_hrsc2016.py \
--type OBB --output_path ./work_dirs/IOUfit_ReDet_re50_refpn_3x_hrsc2016_9_rerun

python DOTA_devkit/hrsc2016_evaluation.py --det_root ./work_dirs/IOUfit_ReDet_re50_refpn_3x_hrsc2016_9_rerun