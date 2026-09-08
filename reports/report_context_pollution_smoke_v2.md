# E1-v2 Corrected Context-Pollution Report

## Overall
category,condition,k_irrelevant,num_samples_per_seed,num_seeds,accuracy_invalid_as_wrong_mean,accuracy_invalid_as_wrong_std,valid_only_accuracy_mean,valid_only_accuracy_std,invalid_rate_mean,invalid_rate_std,mean_input_tokens_mean,mean_input_tokens_std,mean_output_tokens_mean,mean_output_tokens_std,finish_length_rate_mean,finish_length_rate_std,finish_eos_rate_mean,finish_eos_rate_std
ALL,original_only,-1,20,1,0.65,0.0,1.0,0.0,0.35,0.0,7217.153846153846,0.0,2.0,0.0,0.0,0.0,0.65,0.0
ALL,gt_crop_only,0,20,1,0.65,0.0,1.0,0.0,0.35,0.0,7408.153846153846,0.0,2.0,0.0,0.0,0.0,0.65,0.0
ALL,gt_plus_8_irrelevant,8,20,1,0.45,0.0,1.0,0.0,0.55,0.0,5816.888888888889,0.0,2.0,0.0,0.0,0.0,0.45,0.0

## By Category
category,condition,k_irrelevant,num_samples_per_seed,num_seeds,accuracy_invalid_as_wrong_mean,accuracy_invalid_as_wrong_std,valid_only_accuracy_mean,valid_only_accuracy_std,invalid_rate_mean,invalid_rate_std,mean_input_tokens_mean,mean_input_tokens_std,mean_output_tokens_mean,mean_output_tokens_std,finish_length_rate_mean,finish_length_rate_std,finish_eos_rate_mean,finish_eos_rate_std
GPT4V-hard,original_only,-1,17,1,0.5882352941176471,0.0,1.0,0.0,0.4117647058823529,0.0,8043.7,0.0,2.0,0.0,0.0,0.0,0.5882352941176471,0.0
GPT4V-hard,gt_crop_only,0,17,1,0.5882352941176471,0.0,1.0,0.0,0.4117647058823529,0.0,8241.0,0.0,2.0,0.0,0.0,0.0,0.5882352941176471,0.0
GPT4V-hard,gt_plus_8_irrelevant,8,17,1,0.35294117647058826,0.0,1.0,0.0,0.6470588235294117,0.0,5754.333333333333,0.0,2.0,0.0,0.0,0.0,0.35294117647058826,0.0
OCR,original_only,-1,3,1,1.0,0.0,1.0,0.0,0.0,0.0,4462.0,0.0,2.0,0.0,0.0,0.0,1.0,0.0
OCR,gt_crop_only,0,3,1,1.0,0.0,1.0,0.0,0.0,0.0,4632.0,0.0,2.0,0.0,0.0,0.0,1.0,0.0
OCR,gt_plus_8_irrelevant,8,3,1,1.0,0.0,1.0,0.0,0.0,0.0,5942.0,0.0,2.0,0.0,0.0,0.0,1.0,0.0

## Bootstrap
comparison,mode,n_paired_seed_samples,accuracy_gt_crop_only,accuracy_target_condition,mean_difference_gt_minus_target,ci95_low,ci95_high,p_one_sided_diff_gt_0,p_two_sided_diff_ne_0
gt_crop_only_vs_gt_plus_1_irrelevant,invalid_as_wrong,0,,,,,,0.000999000999000999,0.001998001998001998
gt_crop_only_vs_gt_plus_4_irrelevant,invalid_as_wrong,0,,,,,,0.000999000999000999,0.001998001998001998
gt_crop_only_vs_gt_plus_8_irrelevant,invalid_as_wrong,20,0.65,0.45,0.2,0.04999999999999993,0.4,0.016983016983016984,0.03396603396603397
gt_crop_only_vs_gt_plus_1_irrelevant,valid_only,0,,,,,,0.000999000999000999,0.001998001998001998
gt_crop_only_vs_gt_plus_4_irrelevant,valid_only,0,,,,,,0.000999000999000999,0.001998001998001998
gt_crop_only_vs_gt_plus_8_irrelevant,valid_only,9,1.0,1.0,0.0,0.0,0.0,1.0,1.0

## Contact-Sheet Comparison
旧 contact-sheet 的 GT-only 到 +8 drop 为 0.0630；v2 separate-image 的 invalid-as-wrong drop 为 0.2000。