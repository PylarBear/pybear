

def gmlr_default_config_params():

    gmlr_conv_kill = 1
    gmlr_pct_change = 0
    gmlr_conv_end_method = 'KILL'
    gmlr_rglztn_type = 'RIDGE'
    gmlr_rglztn_fctr = 1000
    gmlr_batch_method = 'B'
    gmlr_batch_size = int(1e12)
    gmlr_type = 'F'
    gmlr_score_method = 'Q'
    gmlr_float_only = False
    gmlr_max_columns = float('inf')
    gmlr_bypass_agg = False
    GMLR_OUTPUT_VECTOR = []

    return gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method, gmlr_rglztn_type, gmlr_rglztn_fctr, gmlr_batch_method, \
        gmlr_batch_size, gmlr_type, gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg, GMLR_OUTPUT_VECTOR



