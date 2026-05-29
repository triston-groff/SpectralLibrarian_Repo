import re
import pandas as pd


def extract_ionization(instrument_type):
    if pd.isna(instrument_type):
        return None  # or handle as needed, e.g., 'Unknown'

    patterns = ['ESI', 'MALDI', 'APCI', 'LC', 'DESI']
    for pat in patterns:
        if re.search(pat, instrument_type):
            return pat

    return f"Unknown_{instrument_type}"


# Example usage with the provided mappings
instrument_type_mappings = {
    # QTOF
    'ESI-QTOF': 'QTOF',
    'LC-ESI-ITTOF': 'QTOF',
    'LC-ESI-QTOF': 'QTOF',
    'LC-ESI-TOF': 'QTOF',
    'LC-Q-TOF/MS': 'QTOF',
    'LC-QTOF': 'QTOF',
    'Q-TOF': 'QTOF',
    'QTOF': 'QTOF',
    'SYNAPT QTOF, Waters': 'QTOF',
    'Waters SYNAPT': 'QTOF',

    # Fourier Transform
    # FT-ICR
    'FT-ICR/FTMS': 'FTMS/ICR',
    # Orbitrap
    'LC-ESI-Orbitrap': 'FTMS/Orbitrap',
    'LC-ESI-Q-Orbitrap': 'FTMS/Orbitrap',
    'LC-ESI-QEHF': 'FTMS/Orbitrap',
    'Orbitrap': 'FTMS/Orbitrap',
    'Q Exactive Focus Hybrid Quadrupole Orbitrap Mass Spectrometer (Thermo Fisher Scientific)': 'FTMS/Orbitrap',
    'Q Exactive HF': 'FTMS/Orbitrap',
    # Unknow FTMS
    'ESI-FT': 'FTMS',
    'ESI-ITFT': 'FTMS',
    'ESI-QFT': 'FTMS',
    'IT-FT/FTMS': 'FTMS',
    'LC-ESI-HRMS-FT': 'FTMS',
    'LC-ESI-ITFT': 'FTMS',
    'LC-ESI-QFT': 'FTMS',
    'QIT-FT': 'FTMS',

    # Miscellaneous
    'ESI': 'ESI',
    'LC-ESI-HRMS': 'HRMS',
    'in source CID': 'in source CID',

    # Drop (low res/ionization)
    'APCI': None,
    'EBEQ': None,
    'LC-APPI-QQ': None,
    'LC-ESI-IT': None,
    'LC-ESI-Q': None,
    'LC-ESI-QQ': None,
    'LC-ESI-QQQ': None,
    'LIT': None,
    'Linear Ion Trap': None,
    'QIT': None,
    'QqQ': None,
    'Thermo LTQ': None,
    'APCI-ITFT': None,
    'LC-ESI-QIT': None,
    'APCI-ITTOF': None,
    'MALDI-QITTOF': None,
    'LC-APCI-ITFT': None
}

ionization_mapping = {
    'APCI': None,
    'ESI': 'ESI'
}


def apply_instrument_corrections(dfs):
    """
    Applies corrections to 'INSTRUMENT', 'INSTRUMENTTYPE', and 'IONIZATION' columns across multiple DataFrames
    based on unique combinations from the provided mapping. Updates are done only for exact matching combinations.
    Drops rows where any corrected value contains 'unknown' (case-insensitive).
    Modifications are done in-place.

    Parameters:
    - dfs: List of pandas DataFrames to update (e.g., [spx.df, msn_lib.df, mona.df, nist23.df]).
    """
    mapping = {
        ('Thermo Finnigan Elite Orbitrap', 'FTMS', 'ESI'): ('Thermo Finnigan Elite Orbitrap', 'FTMS/Orbitrap', 'ESI'),
        ('Orbitrap ID-X', 'FTMS/Orbitrap', 'ESI'): ('Thermo Orbitrap ID-X', 'FTMS/Orbitrap', 'ESI'),
        ('Orbitrap Fusion Lumos', 'FTMS', 'ESI'): ('Thermo Orbitrap Fusion Lumos', 'FTMS/Orbitrap', 'ESI'),
        ('Thermo Q Exactive HF', 'FTMS', 'ESI'): ('Thermo Q Exactive HF', 'FTMS/Orbitrap', 'ESI'),
        ('Agilent QTOF 6530', 'QTOF', 'ESI'): ('Agilent QTOF 6530', 'QTOF', 'ESI'),
        ('LTQ Orbitrap XL Thermo Scientific', 'FTMS', 'ESI'): ('Thermo LTQ Orbitrap XL', 'FTMS/Orbitrap', 'ESI'),
        ('Agilent 6530 Q-TOF', 'QTOF', 'ESI'): ('Agilent 6530 QTOF', 'QTOF', 'ESI'),
        ('Thermo Finnigan Velos Orbitrap', 'FTMS', 'ESI'): ('Thermo Finnigan Velos Orbitrap', 'FTMS/Orbitrap', 'ESI'),
        ('Waters Xevo G2 Q-Tof', 'QTOF', 'ESI'): ('Waters Xevo G2 QTOF', 'QTOF', 'ESI'),
        ('Q Exactive Orbitrap (Thermo Scientific)', 'FTMS', 'ESI'): ('Thermo Q Exactive Orbitrap', 'FTMS/Orbitrap',
                                                                     'ESI'),
        ('Agilent qTOF 6545', 'QTOF', 'ESI'): ('Agilent QTOF 6545', 'QTOF', 'ESI'),
        ('Bruker maXis Impact', 'QTOF', 'ESI'): ('Bruker maXis Impact', 'QTOF', 'ESI'),
        ('Q Exactive Plus Orbitrap Thermo Scientific', 'FTMS', 'ESI'): ('Thermo Q Exactive Orbitrap Plus',
                                                                        'FTMS/Orbitrap', 'ESI'),
        ('Agilent 6550 iFunnel', 'QTOF', 'ESI'): ('Agilent 6550 iFunnel', 'QTOF', 'ESI'),
        ('Q Exactive Orbitrap Thermo Scientific', 'FTMS', 'ESI'): ('Thermo Q Exactive Orbitrap', 'FTMS/Orbitrap',
                                                                   'ESI'),
        ('LC, Waters Acquity UPLC System; MS, Waters Xevo G2 Q-Tof', 'QTOF', 'ESI'): ('Waters Xevo G2 QTOF', 'QTOF',
                                                                                      'ESI'),
        ('LC-10ADVPmicro HPLC, Shimadzu; LTQ Orbitrap, Thermo Scientific', 'FTMS', 'ESI'): ('Thermo LTQ Orbitrap',
                                                                                            'FTMS/Orbitrap', 'ESI'),
        ('SCIEX TripleTOF 6600', 'QTOF', 'ESI'): ('Sciex TripleTOF 6600', 'QTOF', 'ESI'),
        ('Thermo Q-Exactive Plus', 'FTMS', 'ESI'): ('Thermo Q Exactive Plus', 'FTMS/Orbitrap', 'ESI'),
        ('Sciex ZenoTOF 8600', 'QTOF', 'ESI'): ('Sciex ZenoTOF 8600', 'QTOF', 'ESI'),
        ('Agilent 1200 RRLC; Agilent 6520 QTOF', 'QTOF', 'ESI'): ('Agilent 6520 QTOF', 'QTOF', 'ESI'),
        ('UPLC Q-Tof Premier, Waters', 'QTOF', 'ESI'): ('Waters QTOF Premier', 'QTOF', 'ESI'),
        ('Q-Exactive Orbitrap Thermo Scientific', 'FTMS', 'ESI'): ('Thermo Q Exactive Orbitrap', 'FTMS/Orbitrap',
                                                                   'ESI'),
        ('maXis plus UHR-ToF-MS, Bruker Daltonics', 'QTOF', 'ESI'): ('Bruker maXis Plus', 'QTOF', 'ESI'),
        ('Q-Tof Premier, Waters', 'QTOF', 'LC'): ('Waters QTOF Premier', 'QTOF', 'LC'),
        ('maXis (Bruker Daltonics)', 'QTOF', 'ESI'): ('Bruker maXis', 'QTOF', 'ESI'),
        ('AB Sciex TripleTOF 5600+ system (Q-TOF) equipped with a DuoSpray ion source', 'QTOF', 'ESI'): (
            'Sciex TripleTOF 5600+', 'QTOF', 'ESI'),
        ('impact HD', 'QTOF', 'ESI'): ('impact HD', 'QTOF', 'ESI'),
        ('LTQ Orbitrap XL, Thermo Scientfic; HP-1100 HPLC, Agilent', 'FTMS', 'ESI'): ('Thermo LTQ Orbitrap XL',
                                                                                      'FTMS/Orbitrap', 'ESI'),
        ('Agilent QTOF 6550', 'QTOF', 'ESI'): ('Agilent QTOF 6550', 'QTOF', 'ESI'),
        ('Thermo Scientific Orbitrap ID-X', 'FTMS/Orbitrap', 'ESI'): ('Thermo Orbitrap ID-X', 'FTMS/Orbitrap', 'ESI'),
        ('Micromass Q-TOF II', 'QTOF', 'ESI'): ('Micromass QTOF II', 'QTOF', 'ESI'),
        ('Sciex TripleTOF5600', 'QTOF', 'LC'): ('Sciex TripleTOF 5600', 'QTOF', 'LC'),
        ('maXis, Bruker Daltonics', 'QTOF', 'ESI'): ('Bruker maXis', 'QTOF', 'ESI'),
        ('API QSTAR Pulsar i', 'QTOF', 'ESI'): ('API QSTAR Pulsar i', 'QTOF', 'ESI'),
        ('Q Exactive Thermo Fisher Scientific', 'FTMS', 'ESI'): ('Thermo Q Exactive', 'FTMS/Orbitrap', 'ESI'),
        ('Thermo QExactive-HF', 'FTMS/Orbitrap', 'ESI'): ('Thermo QExactive-HF', 'FTMS/Orbitrap', 'ESI'),
        ('Orbitrap Fusion, Thermo Scientific.', 'FTMS', 'ESI'): ('Thermo Orbitrap Fusion', 'FTMS/Orbitrap', 'ESI'),
        ('LTQ Orbitrap XL, Thermo Scientfic', 'FTMS', 'ESI'): ('Thermo LTQ Orbitrap XL', 'FTMS/Orbitrap', 'ESI'),
        ('LTQ Orbitrap XL Thermo Fisher Scientific', 'FTMS', 'ESI'): ('Thermo LTQ Orbitrap XL', 'FTMS/Orbitrap', 'ESI'),
        ('Orbitrap Classic, Thermo Scientific', 'FTMS', 'ESI'): ('Thermo Orbitrap Classic', 'FTMS/Orbitrap', 'ESI'),
        ('Q-Exactive HF, Thermo Scientific [MS:1002523]', 'FTMS', 'ESI'): ('Thermo Q Exactive HF', 'FTMS/Orbitrap',
                                                                           'ESI'),
        ('Q-Exactive HF, Thermo Scientific [MS:1002523]', 'QTOF', 'ESI'): ('Thermo Q Exactive HF', 'FTMS/Orbitrap',
                                                                           'ESI'),
        ('micrOTOF-Q', 'QTOF', 'ESI'): ('micrOTOF-Q', 'QTOF', 'ESI'),
        ('Agilent QTof 6545, Agilent Technologies [MS:1000490]', 'QTOF', 'ESI'): ('Agilent QTOF 6545', 'QTOF', 'ESI'),
        ('6550 Q-TOF (Agilent Technologies)', 'QTOF', 'ESI'): ('Agilent 6550 QTOF', 'QTOF', 'ESI'),
        ('Agilent 6530', 'QTOF', 'ESI'): ('Agilent 6530', 'QTOF', 'ESI'),
        ('ESI', 'ESI', 'ESI'): ('ESI', 'ESI', 'ESI'),
        ('Q-Exactive + Thermo Scientific', 'FTMS', 'ESI'): ('Thermo Q Exactive Plus', 'FTMS/Orbitrap', 'ESI'),
        ('Waters SYNAPT-G2 QTOF', 'QTOF', 'ESI'): ('Waters SYNAPT-G2 QTOF', 'QTOF', 'ESI'),
        ('X500R QTOF (AB Sciex LLC, USA)', 'QTOF', 'ESI'): ('Sciex X500R QTOF', 'QTOF', 'ESI'),
        ('Orbitrap Fusion Tribrid Thermo Fisher Scientific', 'FTMS', 'ESI'): ('Thermo Orbitrap Fusion Tribrid',
                                                                              'FTMS/Orbitrap', 'ESI'),
        ('LTQ Orbitrap Velos Thermo Scientific', 'FTMS', 'ESI'): ('Thermo LTQ Orbitrap Velos', 'FTMS/Orbitrap', 'ESI'),
        ('Xevo G2-S QtOF, Waters (USA) coupled to ACQUITY UPLC, Waters (USA).', 'QTOF', 'ESI'): (
            'Waters Xevo G2-S QTOF', 'QTOF', 'ESI'),
        ('QTOF Premier', 'QTOF', 'Unknown_Q-TOF'): ('Waters QTOF Premier', 'QTOF', 'Unknown_Q-TOF'),
        ('LCMS-IT-TOF', 'QTOF', 'ESI'): ('LCMS-IT-TOF', 'QTOF', 'ESI'),
        ('API QSTAR', 'QTOF', 'Unknown_Q-TOF'): ('API QSTAR', 'QTOF', 'Unknown_Q-TOF'),
        ('Bruker MicrOTOF-Q', 'QTOF', 'ESI'): ('Bruker MicrOTOF-Q', 'QTOF', 'ESI'),
        ('6550 QTOF (Agilent Technologies)', 'QTOF', 'ESI'): ('Agilent 6550 QTOF', 'QTOF', 'ESI'),
        ('X500R QTOF (AB Sciex Pte. Ltd, USA)', 'QTOF', 'ESI'): ('Sciex X500R QTOF', 'QTOF', 'ESI'),
        ('Waters/Micromass Q-TOF Ultima', 'QTOF', 'ESI'): ('Waters QTOF Ultima', 'QTOF', 'ESI'),
        ('Thermo Q Exactive Plus', 'FTMS', 'ESI'): ('Thermo Q Exactive Plus', 'FTMS/Orbitrap', 'ESI'),
        ('Bruker micrOTOF-Q', 'QTOF', 'ESI'): ('Bruker MicrOTOF-Q', 'QTOF', 'ESI'),
        ('Thermo Exploris 480', 'FTMS', 'ESI'): ('Thermo Exploris 480', 'FTMS/Orbitrap', 'ESI'),
        ('Q-Exactive Thermo Scientific', 'FTMS', 'ESI'): ('Thermo Q Exactive', 'FTMS/Orbitrap', 'ESI'),
        ('LTQ Orbitrap XL hybrid iontrap-Orbitrap (Thermo Fisher Scientific, San Jose, CA, USA)', 'FTMS', 'ESI'): (
            'Thermo LTQ Orbitrap XL', 'FTMS/Orbitrap', 'ESI'),
        ('QTOF Ultima API', 'QTOF', 'Unknown_Q-TOF'): ('QTOF Ultima API', 'QTOF', 'Unknown_Q-TOF'),
        ('Micromass Quattromicro', 'in source CID', 'Unknown_in source CID'): ('Micromass Quattromicro', 'QQQ',
                                                                               'Unknown_in source CID'),
        ('Micromass Global Ultima', 'QTOF', 'ESI'): ('Micromass Global Ultima', 'QTOF', 'ESI'),
        ('Micromass Q-TOF2', 'QTOF', 'ESI'): ('Micromass QTOF2', 'QTOF', 'ESI'),
        ('Bruker maXis ESI-QTOF', 'QTOF', 'ESI'): ('Bruker maXis', 'QTOF', 'ESI'),
        ('LTQ Orbitap Velos (Thermofisher Scientific)', 'FTMS', 'ESI'): ('Thermo LTQ Orbitap Velos', 'FTMS/Orbitrap',
                                                                         'ESI'),
        ('THERMO Q EXACTIVE PLUS', 'FTMS', 'ESI'): ('Thermo Q Exactive Plus', 'FTMS/Orbitrap', 'ESI'),
        ('Agilent 6560 QTOF', 'QTOF', 'ESI'): ('Agilent 6560 QTOF', 'QTOF', 'ESI'),
        ('home-made, prototype PE Sciex API 365', 'QTOF', 'ESI'): ('PE Sciex API 365 (homemade prototype)', 'QTOF',
                                                                   'ESI'),
        ('LCT Micromass', 'QTOF', 'ESI'): ('LCT Micromass', 'QTOF', 'ESI'),
        ('LCMS-ITTOF, Shimadzu', 'QTOF', 'ESI'): ('LCMS-ITTOF, Shimadzu', 'QTOF', 'ESI'),
        ('Waters QTof-Micro', 'QTOF', 'ESI'): ('Waters QTOF Micro', 'QTOF', 'ESI'),
        ('6599 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6599 QTOF', 'QTOF', 'ESI'),
        ('LTQ Orbitrap XL, Thermo Scientific', 'FTMS', 'ESI'): ('Thermo LTQ Orbitrap XL', 'FTMS/Orbitrap', 'ESI'),
        ('Bruker BioApex II 47e', 'FTMS/ICR', 'ESI'): ('Bruker BioApex II 47e', 'FTMS/ICR', 'ESI'),
        ('Micromass Q-TOF micro', 'QTOF', 'ESI'): ('Micromass QTOF micro', 'QTOF', 'ESI'),
        ('ZQ', 'in source CID', 'Unknown_in source CID'): ('ZQ', 'Q', 'Unknown_in source CID'),
        ('Xevo G2 XS QTOF, waters', 'QTOF', 'ESI'): ('Waters Xevo G2 XS QTOF', 'QTOF', 'ESI'),
        ('Xevo QTOF (Waters)', 'QTOF', 'ESI'): ('Waters Xevo QTOF', 'QTOF', 'ESI'),
        ('6540 Q-TOF Agilent', 'QTOF', 'ESI'): ('Agilent 6540 QTOF', 'QTOF', 'ESI'),
        ('Agilent 6530', 'FTMS/Orbitrap', 'ESI'): ('Agilent 6530 QTOF', 'QTOF', 'ESI'),
        ('Micromass Q-TOF 2', 'QTOF', 'ESI'): ('Micromass QTOF 2', 'QTOF', 'ESI'),
        ('Micromass Q-TOF', 'QTOF', 'ESI'): ('Micromass QTOF', 'QTOF', 'ESI'),
        ('Applied Biosystems API QSTAR Pulsar', 'QTOF', 'ESI'): ('Applied Biosystems API QSTAR Pulsar', 'QTOF', 'ESI'),
        ('Bruker Apex III', 'FTMS/ICR', 'ESI'): ('Bruker Apex III', 'FTMS/ICR', 'ESI'),
        ('9.4 Tesla home-made', 'FTMS/ICR', 'ESI'): ('9.4 Tesla home-made', 'FTMS/ICR', 'ESI'),
        ('PE Sciex QSTAR Pulsar', 'QTOF', 'ESI'): ('PE Sciex QSTAR Pulsar', 'QTOF', 'ESI'),
        ('6535 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6535 QTOF', 'QTOF', 'ESI'),
        ('6536 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6536 QTOF', 'QTOF', 'ESI'),
        ('6541 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6541 QTOF', 'QTOF', 'ESI'),
        ('6538 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6538 QTOF', 'QTOF', 'ESI'),
        ('6539 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6539 QTOF', 'QTOF', 'ESI'),
        ('6543 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6543 QTOF', 'QTOF', 'ESI'),
        ('6542 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6542 QTOF', 'QTOF', 'ESI'),
        ('6534 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6534 QTOF', 'QTOF', 'ESI'),
        ('6537 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6537 QTOF', 'QTOF', 'ESI'),
        ('6540 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6540 QTOF', 'QTOF', 'ESI'),
        ('6546 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6546 QTOF', 'QTOF', 'ESI'),
        ('6549 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6549 QTOF', 'QTOF', 'ESI'),
        ('6548 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6548 QTOF', 'QTOF', 'ESI'),
        ('6550 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6550 QTOF', 'QTOF', 'ESI'),
        ('6551 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6551 QTOF', 'QTOF', 'ESI'),
        ('6552 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6552 QTOF', 'QTOF', 'ESI'),
        ('6553 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6553 QTOF', 'QTOF', 'ESI'),
        ('6554 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6554 QTOF', 'QTOF', 'ESI'),
        ('6555 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6555 QTOF', 'QTOF', 'ESI'),
        ('6556 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6556 QTOF', 'QTOF', 'ESI'),
        ('6557 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6557 QTOF', 'QTOF', 'ESI'),
        ('6558 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6558 QTOF', 'QTOF', 'ESI'),
        ('6559 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6559 QTOF', 'QTOF', 'ESI'),
        ('6544 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6544 QTOF', 'QTOF', 'ESI'),
        ('6545 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6545 QTOF', 'QTOF', 'ESI'),
        ('6547 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6547 QTOF', 'QTOF', 'ESI'),
        ('6533 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6533 QTOF', 'QTOF', 'ESI'),
        ('6531 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6531 QTOF', 'QTOF', 'ESI'),
        ('6530 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6530 QTOF', 'QTOF', 'ESI'),
        ('6631 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6631 QTOF', 'QTOF', 'ESI'),
        ('6630 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6630 QTOF', 'QTOF', 'ESI'),
        ('6629 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6629 QTOF', 'QTOF', 'ESI'),
        ('6628 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6628 QTOF', 'QTOF', 'ESI'),
        ('6627 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6627 QTOF', 'QTOF', 'ESI'),
        ('6626 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6626 QTOF', 'QTOF', 'ESI'),
        ('6625 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6625 QTOF', 'QTOF', 'ESI'),
        ('6624 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6624 QTOF', 'QTOF', 'ESI'),
        ('6623 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6623 QTOF', 'QTOF', 'ESI'),
        ('6622 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6622 QTOF', 'QTOF', 'ESI'),
        ('6621 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6621 QTOF', 'QTOF', 'ESI'),
        ('6620 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6620 QTOF', 'QTOF', 'ESI'),
        ('6619 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6619 QTOF', 'QTOF', 'ESI'),
        ('6618 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6618 QTOF', 'QTOF', 'ESI'),
        ('6617 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6617 QTOF', 'QTOF', 'ESI'),
        ('6616 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6616 QTOF', 'QTOF', 'ESI'),
        ('6615 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6615 QTOF', 'QTOF', 'ESI'),
        ('6614 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6614 QTOF', 'QTOF', 'ESI'),
        ('6613 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6613 QTOF', 'QTOF', 'ESI'),
        ('6612 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6612 QTOF', 'QTOF', 'ESI'),
        ('6611 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6611 QTOF', 'QTOF', 'ESI'),
        ('6610 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6610 QTOF', 'QTOF', 'ESI'),
        ('6609 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6609 QTOF', 'QTOF', 'ESI'),
        ('6608 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6608 QTOF', 'QTOF', 'ESI'),
        ('6607 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6607 QTOF', 'QTOF', 'ESI'),
        ('6606 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6606 QTOF', 'QTOF', 'ESI'),
        ('6605 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6605 QTOF', 'QTOF', 'ESI'),
        ('6604 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6604 QTOF', 'QTOF', 'ESI'),
        ('6603 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6603 QTOF', 'QTOF', 'ESI'),
        ('6602 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6602 QTOF', 'QTOF', 'ESI'),
        ('6601 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6601 QTOF', 'QTOF', 'ESI'),
        ('6598 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6598 QTOF', 'QTOF', 'ESI'),
        ('6581 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6581 QTOF', 'QTOF', 'ESI'),
        ('6597 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6597 QTOF', 'QTOF', 'ESI'),
        ('6596 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6596 QTOF', 'QTOF', 'ESI'),
        ('6595 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6595 QTOF', 'QTOF', 'ESI'),
        ('6594 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6594 QTOF', 'QTOF', 'ESI'),
        ('6593 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6593 QTOF', 'QTOF', 'ESI'),
        ('6592 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6592 QTOF', 'QTOF', 'ESI'),
        ('6591 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6591 QTOF', 'QTOF', 'ESI'),
        ('6589 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6589 QTOF', 'QTOF', 'ESI'),
        ('6588 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6588 QTOF', 'QTOF', 'ESI'),
        ('6587 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6587 QTOF', 'QTOF', 'ESI'),
        ('6586 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6586 QTOF', 'QTOF', 'ESI'),
        ('6585 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6585 QTOF', 'QTOF', 'ESI'),
        ('6584 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6584 QTOF', 'QTOF', 'ESI'),
        ('6583 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6583 QTOF', 'QTOF', 'ESI'),
        ('6582 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6582 QTOF', 'QTOF', 'ESI'),
        ('6573 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6573 QTOF', 'QTOF', 'ESI'),
        ('6580 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6580 QTOF', 'QTOF', 'ESI'),
        ('6579 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6579 QTOF', 'QTOF', 'ESI'),
        ('6578 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6578 QTOF', 'QTOF', 'ESI'),
        ('6577 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6577 QTOF', 'QTOF', 'ESI'),
        ('6576 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6576 QTOF', 'QTOF', 'ESI'),
        ('6575 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6575 QTOF', 'QTOF', 'ESI'),
        ('6574 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6574 QTOF', 'QTOF', 'ESI'),
        ('6569 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6569 QTOF', 'QTOF', 'ESI'),
        ('6572 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6572 QTOF', 'QTOF', 'ESI'),
        ('6571 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6571 QTOF', 'QTOF', 'ESI'),
        ('6570 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6570 QTOF', 'QTOF', 'ESI'),
        ('6567 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6567 QTOF', 'QTOF', 'ESI'),
        ('6568 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6568 QTOF', 'QTOF', 'ESI'),
        ('6566 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6566 QTOF', 'QTOF', 'ESI'),
        ('6565 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6565 QTOF', 'QTOF', 'ESI'),
        ('6532 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6532 QTOF', 'QTOF', 'ESI'),
        ('6564 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6564 QTOF', 'QTOF', 'ESI'),
        ('6563 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6563 QTOF', 'QTOF', 'ESI'),
        ('6562 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6562 QTOF', 'QTOF', 'ESI'),
        ('6561 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6561 QTOF', 'QTOF', 'ESI'),
        ('6560 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6560 QTOF', 'QTOF', 'ESI'),
        ('6680 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6680 QTOF', 'QTOF', 'ESI'),
        ('6671 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6671 QTOF', 'QTOF', 'ESI'),
        ('6672 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6672 QTOF', 'QTOF', 'ESI'),
        ('6673 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6673 QTOF', 'QTOF', 'ESI'),
        ('6674 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6674 QTOF', 'QTOF', 'ESI'),
        ('6675 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6675 QTOF', 'QTOF', 'ESI'),
        ('6676 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6676 QTOF', 'QTOF', 'ESI'),
        ('6677 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6677 QTOF', 'QTOF', 'ESI'),
        ('6678 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6678 QTOF', 'QTOF', 'ESI'),
        ('6679 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6679 QTOF', 'QTOF', 'ESI'),
        ('6664 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6664 QTOF', 'QTOF', 'ESI'),
        ('6665 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6665 QTOF', 'QTOF', 'ESI'),
        ('6666 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6666 QTOF', 'QTOF', 'ESI'),
        ('6667 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6667 QTOF', 'QTOF', 'ESI'),
        ('6668 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6668 QTOF', 'QTOF', 'ESI'),
        ('6669 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6669 QTOF', 'QTOF', 'ESI'),
        ('6670 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6670 QTOF', 'QTOF', 'ESI'),
        ('6655 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6655 QTOF', 'QTOF', 'ESI'),
        ('6656 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6656 QTOF', 'QTOF', 'ESI'),
        ('6657 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6657 QTOF', 'QTOF', 'ESI'),
        ('6658 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6658 QTOF', 'QTOF', 'ESI'),
        ('6659 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6659 QTOF', 'QTOF', 'ESI'),
        ('6660 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6660 QTOF', 'QTOF', 'ESI'),
        ('6661 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6661 QTOF', 'QTOF', 'ESI'),
        ('6662 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6662 QTOF', 'QTOF', 'ESI'),
        ('6663 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6663 QTOF', 'QTOF', 'ESI'),
        ('6647 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6647 QTOF', 'QTOF', 'ESI'),
        ('6646 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6646 QTOF', 'QTOF', 'ESI'),
        ('6645 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6645 QTOF', 'QTOF', 'ESI'),
        ('6644 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6644 QTOF', 'QTOF', 'ESI'),
        ('6643 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6643 QTOF', 'QTOF', 'ESI'),
        ('6642 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6642 QTOF', 'QTOF', 'ESI'),
        ('6641 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6641 QTOF', 'QTOF', 'ESI'),
        ('6640 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6640 QTOF', 'QTOF', 'ESI'),
        ('6639 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6639 QTOF', 'QTOF', 'ESI'),
        ('6638 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6638 QTOF', 'QTOF', 'ESI'),
        ('6637 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6637 QTOF', 'QTOF', 'ESI'),
        ('6636 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6636 QTOF', 'QTOF', 'ESI'),
        ('6635 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6635 QTOF', 'QTOF', 'ESI'),
        ('6634 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6634 QTOF', 'QTOF', 'ESI'),
        ('6633 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6633 QTOF', 'QTOF', 'ESI'),
        ('6648 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6648 QTOF', 'QTOF', 'ESI'),
        ('6632 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6632 QTOF', 'QTOF', 'ESI'),
        ('6649 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6649 QTOF', 'QTOF', 'ESI'),
        ('6650 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6650 QTOF', 'QTOF', 'ESI'),
        ('6651 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6651 QTOF', 'QTOF', 'ESI'),
        ('6652 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6652 QTOF', 'QTOF', 'ESI'),
        ('6653 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6653 QTOF', 'QTOF', 'ESI'),
        ('6654 QTOF Agilent', 'QTOF', 'ESI'): ('Agilent 6654 QTOF', 'QTOF', 'ESI'),
        ('LTQ-FT', 'FTMS', 'Unknown_QIT-FT'): ('LTQ-FT', 'FTMS', 'Unknown_QIT-FT'),
        ('Q Exactive Orbitrap Thermo Scientific', 'HRMS', 'ESI'): ('Thermo Q Exactive Orbitrap', 'FTMS/Orbitrap',
                                                                   'ESI'),
        ('Q Exactive Orbitrap Plus Thermo Scientific', 'FTMS', 'ESI'): ('Thermo Q Exactive Orbitrap Plus',
                                                                        'FTMS/Orbitrap', 'ESI'),
        ('LC-ESI-ITTOF, Shimadzu', 'QTOF', 'ESI'): ('LC-ESI-ITTOF, Shimadzu', 'QTOF', 'ESI'),
        ('TSQ 7000', 'in source CID', 'Unknown_in source CID'): ('Thermo TSQ 7000', 'QQQ', 'Unknown_in source CID'),
        ('Thermo Q Exactive', 'FTMS/Orbitrap', 'ESI'): ('Thermo Q Exactive', 'FTMS/Orbitrap', 'ESI'),
        ('Thermo Q Exactive HF', 'FTMS/Orbitrap', 'Unknown_Orbitrap'): ('Thermo Q Exactive HF', 'FTMS/Orbitrap',
                                                                        'Unknown_Orbitrap'),
        ('X500R QTOF ( AB Sciex LLC, USA)', 'QTOF', 'ESI'): ('Sciex X500R QTOF', 'QTOF', 'ESI'),
        ('Waters Q-TOF SYNAPT', 'QTOF', 'ESI'): ('Waters QTOF SYNAPT', 'QTOF', 'ESI'),
    }

    for df in dfs:
        # Create temporary key column, handling NaNs (skip if any is NaN)
        def make_key(row):
            try:
                if pd.isna(row['INSTRUMENT']) or pd.isna(row['INSTRUMENTTYPE']) or pd.isna(row['IONIZATION']):
                    return None
                return (row['INSTRUMENT'], row['INSTRUMENTTYPE'], row['IONIZATION'])
            except KeyError:
                return None

        df['original_key'] = df.apply(make_key, axis=1)

        # Get indices where key is in mapping
        mask = df['original_key'].isin(mapping.keys())
        to_drop = []

        for idx in df[mask].index:
            orig_key = df.at[idx, 'original_key']
            corr_inst, corr_type, corr_ion = mapping[orig_key]

            # Check if any corrected value contains 'unknown' (case-insensitive)
            if ('unknown' in corr_inst.lower()) or ('unknown' in corr_type.lower()) or ('unknown' in corr_ion.lower()):
                to_drop.append(idx)
            else:
                # Update columns
                df.at[idx, 'INSTRUMENT'] = corr_inst
                df.at[idx, 'INSTRUMENTTYPE'] = corr_type
                df.at[idx, 'IONIZATION'] = corr_ion

        # Drop rows marked for dropping
        df.drop(to_drop, inplace=True)

        # Clean up temp column and reset index
        df.drop(columns=['original_key'], inplace=True)
        df.reset_index(drop=True, inplace=True)