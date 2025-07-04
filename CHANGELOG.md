# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).



## [0.2.0] - 2025-06-30

### Added
- base:
    cast_to_ndarray,
    check_1D_num_sequence,
    check_1D_str_sequence,
    check_2D_num_array,
    check_2D_str_array,
    check_dtype,
    check_feature_names,
    check_n_features,
    check_is_finite,
    check_is_fitted,
    check_scipy_sparse,
    check_shape,
    copy_X,
    ensure_2D,
    get_feature_names,
    get_feature_names_out,
    is_fitted,
    num_features,
    num_samples,
    set_order,
    validate_data,
    FeatureMixin,
    FileDumpMixin,
    FitTransformMixin,
    GetParamsMixin,
    ReprMixin,
    SetOutputMixin,
    SetParamsMixin

- feature_extraction.text:
    AutoTextCleaner,
    Lexicon,
    NGramMerger,
    StopRemover
    TextJoiner,
    TextJustifier,
    TextLookup,
    TextLookupRealTime,
    TextNormalizer,
    TextPadder,
    TextRemover,
    TextReplacer,
    TextSplitter,
    TextStatistics,
    TextStripper

- model_selection:
    autogridsearch_wrapper,
    GSTCV,
    AutoGridSearchCV,
    AutoGSTCV

- new_numpy:
    random.choice,
    random.Sparse
    random.sparse

- preprocessing:
    ColumnDeduplicateTransformer,
    InterceptManager,
    MinCountTransformer,
    NanStandardizer,
    SlimPolyFeatures

- utilities:
    array_sparsity, 
    check_pipeline, 
    get_methods_out_of_class,
    get_module_name,
    permuter,
    serial_index_mapper,
    time_memory_benchmark,
    timer


### Changed
- None

### Deprecated
- None

### Removed
- None

### Fixed
- None

### Security
- None

## [0.1] - [Unreleased]



