# Autogenerated by nbdev

d = { 'settings': { 'branch': 'main',
                'doc_baseurl': '/step',
                'doc_host': 'https://BorjaRequena.github.io',
                'git_url': 'https://github.com/BorjaRequena/step',
                'lib_path': 'step'},
  'syms': { 'step.baselines': {'step.baselines.ruptures_cp': ('source/baselines.html#ruptures_cp', 'step/baselines.py')},
            'step.core': {'step.core.foo': ('core.html#foo', 'step/core.py')},
            'step.data': { 'step.data.SegmentationTransform': ('source/data.html#segmentationtransform', 'step/data.py'),
                           'step.data.SegmentationTransform.__init__': ('source/data.html#segmentationtransform.__init__', 'step/data.py'),
                           'step.data.SegmentationTransform._get_xy': ('source/data.html#segmentationtransform._get_xy', 'step/data.py'),
                           'step.data.SegmentationTransform.encodes': ('source/data.html#segmentationtransform.encodes', 'step/data.py'),
                           'step.data._add_permutation_sample': ('source/data.html#_add_permutation_sample', 'step/data.py'),
                           'step.data._check_in': ('source/data.html#_check_in', 'step/data.py'),
                           'step.data._filter_dataset': ('source/data.html#_filter_dataset', 'step/data.py'),
                           'step.data._get_balanced_dataset_numbers': ('source/data.html#_get_balanced_dataset_numbers', 'step/data.py'),
                           'step.data._get_bmds_fname': ('source/data.html#_get_bmds_fname', 'step/data.py'),
                           'step.data._get_change_points': ('source/data.html#_get_change_points', 'step/data.py'),
                           'step.data._get_dls_from_ds': ('source/data.html#_get_dls_from_ds', 'step/data.py'),
                           'step.data._get_n_class': ('source/data.html#_get_n_class', 'step/data.py'),
                           'step.data._get_segds_fname': ('source/data.html#_get_segds_fname', 'step/data.py'),
                           'step.data._merge_trajectories': ('source/data.html#_merge_trajectories', 'step/data.py'),
                           'step.data._one_hot_encode': ('source/data.html#_one_hot_encode', 'step/data.py'),
                           'step.data._permute_datasets': ('source/data.html#_permute_datasets', 'step/data.py'),
                           'step.data._prepend_initial_token': ('source/data.html#_prepend_initial_token', 'step/data.py'),
                           'step.data._subsample_dataset': ('source/data.html#_subsample_dataset', 'step/data.py'),
                           'step.data._txt2df': ('source/data.html#_txt2df', 'step/data.py'),
                           'step.data.add_localization_noise': ('source/data.html#add_localization_noise', 'step/data.py'),
                           'step.data.brownian_motion': ('source/data.html#brownian_motion', 'step/data.py'),
                           'step.data.combine_datasets': ('source/data.html#combine_datasets', 'step/data.py'),
                           'step.data.combine_trajectories': ('source/data.html#combine_trajectories', 'step/data.py'),
                           'step.data.create_bm_segmentation_dataset': ('source/data.html#create_bm_segmentation_dataset', 'step/data.py'),
                           'step.data.create_bm_trajectories': ('source/data.html#create_bm_trajectories', 'step/data.py'),
                           'step.data.create_fixed_attm_trajs': ('source/data.html#create_fixed_attm_trajs', 'step/data.py'),
                           'step.data.create_segmentation_dataset': ('source/data.html#create_segmentation_dataset', 'step/data.py'),
                           'step.data.create_trajectories': ('source/data.html#create_trajectories', 'step/data.py'),
                           'step.data.get_andi_valid_dls': ('source/data.html#get_andi_valid_dls', 'step/data.py'),
                           'step.data.get_segmentation_dls': ('source/data.html#get_segmentation_dls', 'step/data.py'),
                           'step.data.get_transformer_dls': ('source/data.html#get_transformer_dls', 'step/data.py'),
                           'step.data.load_andi_data': ('source/data.html#load_andi_data', 'step/data.py'),
                           'step.data.load_dataset': ('source/data.html#load_dataset', 'step/data.py'),
                           'step.data.trajs2df': ('source/data.html#trajs2df', 'step/data.py')},
            'step.models': { 'step.models.AttnDynamicUnet': ('source/models.html#attndynamicunet', 'step/models.py'),
                             'step.models.AttnDynamicUnet.__del__': ('source/models.html#attndynamicunet.__del__', 'step/models.py'),
                             'step.models.AttnDynamicUnet.__init__': ('source/models.html#attndynamicunet.__init__', 'step/models.py'),
                             'step.models.Classifier': ('source/models.html#classifier', 'step/models.py'),
                             'step.models.Classifier.__init__': ('source/models.html#classifier.__init__', 'step/models.py'),
                             'step.models.Classifier.forward': ('source/models.html#classifier.forward', 'step/models.py'),
                             'step.models.ConcatPooling': ('source/models.html#concatpooling', 'step/models.py'),
                             'step.models.ConcatPooling.__init__': ('source/models.html#concatpooling.__init__', 'step/models.py'),
                             'step.models.ConcatPooling.forward': ('source/models.html#concatpooling.forward', 'step/models.py'),
                             'step.models.ConvAttn': ('source/models.html#convattn', 'step/models.py'),
                             'step.models.ConvAttn.__init__': ('source/models.html#convattn.__init__', 'step/models.py'),
                             'step.models.ConvAttn._reset_parameters': ('source/models.html#convattn._reset_parameters', 'step/models.py'),
                             'step.models.ConvAttn.forward': ('source/models.html#convattn.forward', 'step/models.py'),
                             'step.models.EncoderClassifier': ('source/models.html#encoderclassifier', 'step/models.py'),
                             'step.models.EncoderClassifier.__init__': ('source/models.html#encoderclassifier.__init__', 'step/models.py'),
                             'step.models.EncoderClassifier._reset_parameters': ( 'source/models.html#encoderclassifier._reset_parameters',
                                                                                  'step/models.py'),
                             'step.models.EncoderClassifier.forward': ('source/models.html#encoderclassifier.forward', 'step/models.py'),
                             'step.models.EncoderClassifier.get_random_mask': ( 'source/models.html#encoderclassifier.get_random_mask',
                                                                                'step/models.py'),
                             'step.models.GeneralPixleShuffle': ('source/models.html#generalpixleshuffle', 'step/models.py'),
                             'step.models.GeneralPixleShuffle.__init__': ( 'source/models.html#generalpixleshuffle.__init__',
                                                                           'step/models.py'),
                             'step.models.GeneralPixleShuffle.forward': ( 'source/models.html#generalpixleshuffle.forward',
                                                                          'step/models.py'),
                             'step.models.LinBnDropTrp': ('source/models.html#linbndroptrp', 'step/models.py'),
                             'step.models.LinBnDropTrp.__init__': ('source/models.html#linbndroptrp.__init__', 'step/models.py'),
                             'step.models.Normalization': ('source/models.html#normalization', 'step/models.py'),
                             'step.models.Normalization.__init__': ('source/models.html#normalization.__init__', 'step/models.py'),
                             'step.models.Normalization.forward': ('source/models.html#normalization.forward', 'step/models.py'),
                             'step.models.PixelShuffleUpsampling': ('source/models.html#pixelshuffleupsampling', 'step/models.py'),
                             'step.models.PixelShuffleUpsampling.__init__': ( 'source/models.html#pixelshuffleupsampling.__init__',
                                                                              'step/models.py'),
                             'step.models.PositionalEncoding': ('source/models.html#positionalencoding', 'step/models.py'),
                             'step.models.PositionalEncoding.__init__': ( 'source/models.html#positionalencoding.__init__',
                                                                          'step/models.py'),
                             'step.models.PositionalEncoding.forward': ('source/models.html#positionalencoding.forward', 'step/models.py'),
                             'step.models.ResizeToOrig': ('source/models.html#resizetoorig', 'step/models.py'),
                             'step.models.ResizeToOrig.__init__': ('source/models.html#resizetoorig.__init__', 'step/models.py'),
                             'step.models.ResizeToOrig.forward': ('source/models.html#resizetoorig.forward', 'step/models.py'),
                             'step.models.Transformer': ('source/models.html#transformer', 'step/models.py'),
                             'step.models.Transformer.__init__': ('source/models.html#transformer.__init__', 'step/models.py'),
                             'step.models.Transformer._check_rank': ('source/models.html#transformer._check_rank', 'step/models.py'),
                             'step.models.Transformer._reset_parameters': ( 'source/models.html#transformer._reset_parameters',
                                                                            'step/models.py'),
                             'step.models.Transformer.forward': ('source/models.html#transformer.forward', 'step/models.py'),
                             'step.models.Transformer.get_square_subsequent_mask': ( 'source/models.html#transformer.get_square_subsequent_mask',
                                                                                     'step/models.py'),
                             'step.models.Transformer.segment': ('source/models.html#transformer.segment', 'step/models.py'),
                             'step.models.Transpose': ('source/models.html#transpose', 'step/models.py'),
                             'step.models.Transpose.__init__': ('source/models.html#transpose.__init__', 'step/models.py'),
                             'step.models.Transpose.forward': ('source/models.html#transpose.forward', 'step/models.py'),
                             'step.models.UnetBlock': ('source/models.html#unetblock', 'step/models.py'),
                             'step.models.UnetBlock.__init__': ('source/models.html#unetblock.__init__', 'step/models.py'),
                             'step.models.UnetBlock.forward': ('source/models.html#unetblock.forward', 'step/models.py'),
                             'step.models.UnetModel': ('source/models.html#unetmodel', 'step/models.py'),
                             'step.models.UnetModel.__init__': ('source/models.html#unetmodel.__init__', 'step/models.py'),
                             'step.models.XResAttn': ('source/models.html#xresattn', 'step/models.py'),
                             'step.models.XResAttn.__init__': ('source/models.html#xresattn.__init__', 'step/models.py'),
                             'step.models.XResAttn._reset_parameters': ('source/models.html#xresattn._reset_parameters', 'step/models.py'),
                             'step.models.XResAttn.forward': ('source/models.html#xresattn.forward', 'step/models.py'),
                             'step.models.XResBlocks': ('source/models.html#xresblocks', 'step/models.py'),
                             'step.models.XResBlocks.__init__': ('source/models.html#xresblocks.__init__', 'step/models.py'),
                             'step.models.XResBlocks._make_blocks': ('source/models.html#xresblocks._make_blocks', 'step/models.py'),
                             'step.models.XResBlocks._make_layer': ('source/models.html#xresblocks._make_layer', 'step/models.py'),
                             'step.models._get_acts': ('source/models.html#_get_acts', 'step/models.py'),
                             'step.models._get_sz_change_idxs': ('source/models.html#_get_sz_change_idxs', 'step/models.py'),
                             'step.models.dummy_eval': ('source/models.html#dummy_eval', 'step/models.py'),
                             'step.models.get_act': ('source/models.html#get_act', 'step/models.py'),
                             'step.models.icnr_init_general': ('source/models.html#icnr_init_general', 'step/models.py'),
                             'step.models.in_channels': ('source/models.html#in_channels', 'step/models.py'),
                             'step.models.model_sizes': ('source/models.html#model_sizes', 'step/models.py'),
                             'step.models.tfm_encoder': ('source/models.html#tfm_encoder', 'step/models.py')},
            'step.utils': { 'step.utils._can_merge': ('source/utils.html#_can_merge', 'step/utils.py'),
                            'step.utils._find_split_sizes': ('source/utils.html#_find_split_sizes', 'step/utils.py'),
                            'step.utils._merge_contiguous_values': ('source/utils.html#_merge_contiguous_values', 'step/utils.py'),
                            'step.utils._merge_edge': ('source/utils.html#_merge_edge', 'step/utils.py'),
                            'step.utils._merge_left': ('source/utils.html#_merge_left', 'step/utils.py'),
                            'step.utils._merge_left_or_right': ('source/utils.html#_merge_left_or_right', 'step/utils.py'),
                            'step.utils._merge_right': ('source/utils.html#_merge_right', 'step/utils.py'),
                            'step.utils._merge_splits': ('source/utils.html#_merge_splits', 'step/utils.py'),
                            'step.utils.abundance': ('source/utils.html#abundance', 'step/utils.py'),
                            'step.utils.anomalous_exponent_tmsd': ('source/utils.html#anomalous_exponent_tmsd', 'step/utils.py'),
                            'step.utils.assign_changepoints': ('source/utils.html#assign_changepoints', 'step/utils.py'),
                            'step.utils.change_points_from_splits': ('source/utils.html#change_points_from_splits', 'step/utils.py'),
                            'step.utils.diffusion_coefficient_tmsd': ('source/utils.html#diffusion_coefficient_tmsd', 'step/utils.py'),
                            'step.utils.eval_andi_metrics': ('source/utils.html#eval_andi_metrics', 'step/utils.py'),
                            'step.utils.evaluate_cp_prediction': ('source/utils.html#evaluate_cp_prediction', 'step/utils.py'),
                            'step.utils.find_change_points': ('source/utils.html#find_change_points', 'step/utils.py'),
                            'step.utils.fit_segments': ('source/utils.html#fit_segments', 'step/utils.py'),
                            'step.utils.get_split_classes': ('source/utils.html#get_split_classes', 'step/utils.py'),
                            'step.utils.get_splits': ('source/utils.html#get_splits', 'step/utils.py'),
                            'step.utils.jaccard_index': ('source/utils.html#jaccard_index', 'step/utils.py'),
                            'step.utils.lengths_from_cps': ('source/utils.html#lengths_from_cps', 'step/utils.py'),
                            'step.utils.majority_vote': ('source/utils.html#majority_vote', 'step/utils.py'),
                            'step.utils.post_process_prediction': ('source/utils.html#post_process_prediction', 'step/utils.py'),
                            'step.utils.split_tensor': ('source/utils.html#split_tensor', 'step/utils.py'),
                            'step.utils.tmsd': ('source/utils.html#tmsd', 'step/utils.py'),
                            'step.utils.validate_andi_1': ('source/utils.html#validate_andi_1', 'step/utils.py'),
                            'step.utils.validate_andi_3_alpha': ('source/utils.html#validate_andi_3_alpha', 'step/utils.py'),
                            'step.utils.validate_andi_3_models': ('source/utils.html#validate_andi_3_models', 'step/utils.py')}}}
