####### Constellation mapping table (LUT)
def IQ_LUT(const):
    mapping_table_QAM16 = {
        (0,0,0,0) : -3-3j,
        (0,0,0,1) : -3-1j,
        (0,0,1,0) : -3+3j,
        (0,0,1,1) : -3+1j,
        (0,1,0,0) : -1-3j,
        (0,1,0,1) : -1-1j,
        (0,1,1,0) : -1+3j,
        (0,1,1,1) : -1+1j,
        (1,0,0,0) :  3-3j,
        (1,0,0,1) :  3-1j,
        (1,0,1,0) :  3+3j,
        (1,0,1,1) :  3+1j,
        (1,1,0,0) :  1-3j,
        (1,1,0,1) :  1-1j,
        (1,1,1,0) :  1+3j,
        (1,1,1,1) :  1+1j
    }
    demapping_table_QAM16 = {v : k for k, v in mapping_table_QAM16.items()}

    if const=='16QAM':
        mapping_table = mapping_table_QAM16
        demapping_table = demapping_table_QAM16

    return mapping_table, demapping_table