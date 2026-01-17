#!/bin/bash
# Auto-generated evaluation script
# Generated from: /mnt/mnt/public/zhangruize/MAS/repo/Search-R1/eval_widesearch/result/20260112_164111
# Converted files: 800
# Unique instances: 200
# Trial num: 4

cd /mnt/mnt/public/zhangruize/MAS/repo/WideSearch && \
python scripts/run_infer_and_eval_batching.py \
  --stage eval \
  --response_root /mnt/mnt/public/zhangruize/MAS/repo/Search-R1/eval_widesearch/result/searchr1_qwen_3b/widesearch_format \
  --result_save_root /mnt/mnt/public/zhangruize/MAS/repo/Search-R1/eval_widesearch/result/searchr1_qwen_3b/widesearch_format/eval_results \
  --instance_id ws_en_001,ws_en_002,ws_en_003,ws_en_004,ws_en_005,ws_en_006,ws_en_007,ws_en_008,ws_en_009,ws_en_010,ws_en_011,ws_en_012,ws_en_013,ws_en_014,ws_en_015,ws_en_016,ws_en_017,ws_en_018,ws_en_019,ws_en_020,ws_en_021,ws_en_022,ws_en_023,ws_en_024,ws_en_025,ws_en_026,ws_en_027,ws_en_028,ws_en_029,ws_en_030,ws_en_031,ws_en_032,ws_en_033,ws_en_034,ws_en_035,ws_en_036,ws_en_037,ws_en_038,ws_en_039,ws_en_040,ws_en_041,ws_en_042,ws_en_043,ws_en_044,ws_en_045,ws_en_046,ws_en_047,ws_en_048,ws_en_049,ws_en_050,ws_en_051,ws_en_052,ws_en_053,ws_en_054,ws_en_055,ws_en_056,ws_en_057,ws_en_058,ws_en_059,ws_en_060,ws_en_061,ws_en_062,ws_en_063,ws_en_064,ws_en_065,ws_en_066,ws_en_067,ws_en_068,ws_en_069,ws_en_070,ws_en_071,ws_en_072,ws_en_073,ws_en_074,ws_en_075,ws_en_076,ws_en_077,ws_en_078,ws_en_079,ws_en_080,ws_en_081,ws_en_082,ws_en_083,ws_en_084,ws_en_085,ws_en_086,ws_en_087,ws_en_088,ws_en_089,ws_en_090,ws_en_091,ws_en_092,ws_en_093,ws_en_094,ws_en_095,ws_en_096,ws_en_097,ws_en_098,ws_en_099,ws_en_100,ws_zh_001,ws_zh_002,ws_zh_003,ws_zh_004,ws_zh_005,ws_zh_006,ws_zh_007,ws_zh_008,ws_zh_009,ws_zh_010,ws_zh_011,ws_zh_012,ws_zh_013,ws_zh_014,ws_zh_015,ws_zh_016,ws_zh_017,ws_zh_018,ws_zh_019,ws_zh_020,ws_zh_021,ws_zh_022,ws_zh_023,ws_zh_024,ws_zh_025,ws_zh_026,ws_zh_027,ws_zh_028,ws_zh_029,ws_zh_030,ws_zh_031,ws_zh_032,ws_zh_033,ws_zh_034,ws_zh_035,ws_zh_036,ws_zh_037,ws_zh_038,ws_zh_039,ws_zh_040,ws_zh_041,ws_zh_042,ws_zh_043,ws_zh_044,ws_zh_045,ws_zh_046,ws_zh_047,ws_zh_048,ws_zh_049,ws_zh_050,ws_zh_051,ws_zh_052,ws_zh_053,ws_zh_054,ws_zh_055,ws_zh_056,ws_zh_057,ws_zh_058,ws_zh_059,ws_zh_060,ws_zh_061,ws_zh_062,ws_zh_063,ws_zh_064,ws_zh_065,ws_zh_066,ws_zh_067,ws_zh_068,ws_zh_069,ws_zh_070,ws_zh_071,ws_zh_072,ws_zh_073,ws_zh_074,ws_zh_075,ws_zh_076,ws_zh_077,ws_zh_078,ws_zh_079,ws_zh_080,ws_zh_081,ws_zh_082,ws_zh_083,ws_zh_084,ws_zh_085,ws_zh_086,ws_zh_087,ws_zh_088,ws_zh_089,ws_zh_090,ws_zh_091,ws_zh_092,ws_zh_093,ws_zh_094,ws_zh_095,ws_zh_096,ws_zh_097,ws_zh_098,ws_zh_099,ws_zh_100 \
  --model_config_name SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.2 \
  --trial_num 4
