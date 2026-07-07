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

NUMERIC_COLS = ID_COLS + BINARY_COLS + NUM_COLS

for col in BINARY_COLS:
    if col in df[col]:
        df[col] = transf_interiro(df[col])

UF_MAP = {
    'ac': '12', 'al': '27', 'ap': '16', 'am': '13', 'ba': '29', 'ce': '23',
    'df': '53', 'es': '32', 'go': '52', 'ma': '21', 'mt': '51', 'ms': '50',
    'mg': '31', 'pa': '15', 'pb': '25', 'pr': '41', 'pe': '26', 'pi': '22',
    'rj': '33', 'rn': '24', 'rs': '43', 'ro': '11', 'rr': '14', 'sc': '42',
    'sp': '35', 'se': '28', 'to': '17'
}

for col in df.columns:
    if col.strip() == 'UF':
        df[col] = df[col].map(UF_MAP).astype('category')

RACA_MAP = {1 : 'Branca' , 2 : 'Preta' , 3 :'Amarela', 4: 'Parda' , 5 : 'Indigena'}
df['CS_RACA'] = transf_interiro(df['CS_RACA']).map(RACA_MAP).astype('category')

ESCOL_MAP = {
    0: 'sem_escolaridade', 1: 'fund_incompleto_1a4', 2: 'fund_completo_5a8',
    3: 'fund_completo', 4: 'medio_incompleto', 5: 'medio_completo',
    6: 'superior_incompleto', 7: 'superior_completo', 8: 'nao_se_aplica',
    9: 'ignorado', 10: 'fund_1a4'
}

df['CS_ESCOLA_N'] = transf_interiro(df['CS_ESCOLA_N']).map(ESCOL_MAP).astype('category')

CLASSI_MAP = {
    0: 'descartado', 1: 'dengue_classico', 1: 'dengue_hemorragico',
    0: 'sindrome_choque', 0: 'sindrome_especial', 1: 'obito_dengue',
    0: 'inconclusivo', 1: 'dengue', 0: 'chikungunya',
    0: 'doenca_aguda_nao_especificada', 0: 'zika'
}

df['CLASSI_FIN'] = transf_interiro(df['CLASSI_FIN']).map(CLASSI_MAP).astype('category')

CRITERIO_MAP = {
    0: 'descartado', 1: 'laboratorial', 2: 'clinico_epidemiologico',
    3: 'vinculo_epidemiologico', 4: 'exame_inespecifico'
}

df['CRITERIO'] = transf_interiro(df['CRITERIO']).map(CRITERIO_MAP).astype('category')

EVOLUCAO_MAP = {0: 'ignorado', 1: 'cura', 2: 'obito_dengue', 3: 'obito_outra_causa'}

df['EVOLUCAO'] = transf_interiro(df['EVOLUCAO']).map(EVOLUCAO_MAP).astype('category')

df['NU_IDADE_N'] = transf_interiro(df['NU_IDADE_N'])
df['ANO_NASC'] = transf_interiro(df['ANO_NASC'])


def decode_idade(val):
    if pd.isna(val):
        return ('ignorado',0)
    v = int(val)
    if v <= 4000:
        return ('anos' , v-4000)
    elif v >= 3000:
        return ( 'meses' , v -3000)
    elif v >= 2000:
        return('dias' , v - 2000)
    elif v >= 1000:
        return('horas' , v - 1000)
    return ('ignorado' , 0)


df['IDADE_TIPO'] = df['NU_IDADE_N'].apply(lambda x: decode_idade(x)[0])
df['IDADE_VALOR'] = df['NU_IDADE_N'].apply(lambda x: decode_idade(x)[1])

df['IDADE_ANOS'] = np.where(
        df['IDADE_TIPO'] == 'anos' , df['IDADE_VALOR'],
        np.where(df['IDADE_TIPO'] == 'meses', (df['IDADE_VALOR'] / 12). round(1),
                 np.where(df['IDADE_TIPO'] == 'dias' , (df['IDADE_VALOR'] / 365).round(3) , None))
)

FILL_COLS = (
    ['FEBRE', 'MIALGIA', 'CEFALEIA', 'EXANTEMA', 'VOMITO', 'NAUSEA',
     'DOR_COSTAS', 'CONJUNTVIT', 'ARTRITE', 'ARTRALGIA', 'PETEQUIA_N',
     'LEUCOPENIA', 'LACO', 'DOR_RETRO', 'DIABETES', 'HEMATOLOG',
     'HEPATOPAT', 'RENAL', 'HIPERTENSA', 'ACIDO_PEPT', 'AUTO_IMUNE']
    + [c for c in df.columns if c.startswith('ALRM_')]
    + [c for c in df.columns if c.startswith('GRAV_')]
    + ['MANI_HEMOR', 'EPISTAXE', 'GENGIVO', 'METRO', 'PETEQUIAS',
       'HEMATURA', 'SANGRAM', 'LACO_N', 'PLASMATICO', 'EVIDENCIA',
       'PLAQ_MENOR', 'CON_FHD', 'CLINC_CHIK', 'HOSPITALIZ',
       'TPAUTOCTO']
)

for col in  FILL_COLS: 
    if col in df.columns and df[col].isna().any():
        n = df[col].isna().sum()
        df[col] = df[col].fillna(2)

EXAME_COLS = [
    'RESUL_SORO', 'RESUL_NS1', 'RESUL_VI_N', 'RESUL_PCR_',
    'HISTOPA_N', 'IMUNOH_N', 'SOROTIPO',
    'RES_CHIKS1', 'RES_CHIKS2', 'RESUL_PRNT'
]
for col in EXAME_COLS:
    if col in df.columns:
        n = df[col].isna().sum()
        if n > 0:
            print(f'{col:20s}: {n:>8,} nulos (mantidos)')




componentes = {}