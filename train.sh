bash ./tools/dist_train.sh configs/IOUFit_DOTA/IOUfit_ReDet_re50_refpn_1x_dota1_ms_trainval.py 2

python tools/test.py configs/IOUFit_DOTA/IOUfit_ReDet_re50_refpn_1x_dota1_ms_trainval.py \
    work_dirs/IOUfit_ReDet_re50_refpn_1x_dota1_ms_trainval_4/epoch_12.pth \
    --out work_dirs/IOUfit_ReDet_re50_refpn_1x_dota1_ms_trainval_4/results.pkl

python tools/parse_results.py --config configs/IOUFit_DOTA/IOUfit_ReDet_re50_refpn_1x_dota1_ms_trainval.py \
--type OBB --output_path ./work_dirs/IOUfit_ReDet_re50_refpn_1x_dota1_ms_trainval_4