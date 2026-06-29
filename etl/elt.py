import pandas as pd
import numpy as np
import os, warnings

warnings.filterwarnings('ignore')

BASE = os.getcwd()

if BASE.endswith('etl'):
    BASE = os.path.dirname(BASE)

PASTA_DADOS = os.path.join(BASE, 'dados')

df_25 = pd.read_csv(os.path.join(PASTA_DADOS, 'DENGBR25.csv'), low_memory= False)
df_26 = pd.read_csv(os.path.join(PASTA_DADOS, 'DENGBR26.csv'), low_memory= False)


df = pd.concat([df_25, df_26], ignore_index= True)

nulos = df.isnull().sum()

# Tratando valores do tipo datetime

DATE_COLS = [
    'DT_NOTIFIC', 'DT_SIN_PRI', 'DT_INVEST', 'DT_SORO', 'DT_NS1',
    'DT_PCR', 'DT_INTERNA', 'DT_ENCERRA', 'DT_ALRM', 'DT_GRAV',
    'DT_DIGITA', 'DT_OBITO','DT_VIRAL','DT_CHIK_S1', 'DT_CHIK_S2', 'DT_PRNT'
]

for col in DATE_COLS:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')


date_cols = [c for c in df.columns if 'DT_' in c and pd.api.types.is_datetime64_any_dtype] 

# esse metodo pd.api.types.is_datetime64_any_dtype faz a seleção das colunas com tipos numericos, diferente do astype(datetime) ele não altera o dataframe apenas faz as consulatas

for col in date_cols:
    validos = df[col].dropna()

    if len(validos) >0:
        print(f'{col:20s} {validos.min().date()} -> {validos.max().date()} ({len(validos):>8,} valdias)')
    else:
        print(f'{col:20s} [todas vazias]')


# Tratando as variaveis numericas

def transf_interiro(dado):
    return pd.to_numeric(dado, errors= 'coerce').astype("Int64")

BINARY_COLS = [
    'FEBRE', 'MIALGIA', 'CEFALEIA', 'EXANTEMA', 'VOMITO', 'NAUSEA',
    'DOR_COSTAS', 'CONJUNTVIT', 'ARTRITE', 'ARTRALGIA', 'PETEQUIA_N',
    'LEUCOPENIA', 'LACO', 'DOR_RETRO', 'DIABETES', 'HEMATOLOG',
    'HEPATOPAT', 'RENAL', 'HIPERTENSA', 'ACIDO_PEPT', 'AUTO_IMUNE',
    'HOSPITALIZ', 'ALRM_HIPOT', 'ALRM_PLAQ', 'ALRM_VOM', 'ALRM_SANG',
    'ALRM_HEMAT', 'ALRM_ABDOM', 'ALRM_LETAR', 'ALRM_HEPAT', 'ALRM_LIQ',
    'GRAV_PULSO', 'GRAV_CONV', 'GRAV_ENCH', 'GRAV_INSUF', 'GRAV_TAQUI',
    'GRAV_EXTRE', 'GRAV_HIPOT', 'GRAV_HEMAT', 'GRAV_MELEN', 'GRAV_METRO',
    'GRAV_SANG', 'GRAV_AST', 'GRAV_MIOC', 'GRAV_CONSC', 'GRAV_ORGAO',
    'MANI_HEMOR', 'EPISTAXE', 'GENGIVO', 'METRO', 'PETEQUIAS',
    'HEMATURA', 'SANGRAM', 'LACO_N', 'PLASMATICO', 'EVIDENCIA',
    'PLAQ_MENOR', 'CON_FHD', 'TPAUTOCTO', 'CLINC_CHIK'
]

NUM_COLS = [
    'RESUL_SORO', 'RESUL_NS1', 'RESUL_VI_N', 'RESUL_PCR_',
    'RES_CHIKS1', 'RES_CHIKS2', 'RESUL_PRNT', 'SOROTIPO',
    'HISTOPA_N', 'IMUNOH_N', 'COMPLICA', 'DOENCA_TRA',
    'TP_NOT', 'CS_FLXRET', 'FLXRECEBI', 'MIGRADO_W',
    'NDUPLIC_N', 'TP_SISTEMA'
]

ID_COLS = [
    'ID_MUNICIP', 'ID_REGIONA', 'ID_UNIDADE', 'ID_MN_RESI',
    'ID_RG_RESI', 'ID_PAIS', 'ID_OCUPA_N', 'MUNICIPIO',
    'COUFINF', 'COPAISINF', 'COMUNINF'
]

NUMERIC_COLS = []

NUMERIC_COLS = NUMERIC_COLS.append([BINARY_COLS,NUM_COLS,ID_COLS])

for col in BINARY_COLS:
    if col in df[col]:
        df[col] = transf_interiro(df[col])