"""Painel de Análise de Dengue.

Análises:
  1. Previsão do número de casos do próximo mês (ARIMA).
  2. Identificação dos municípios mais críticos (clustering).
  3. Priorização e ações recomendadas com base na criticidade.
"""

from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

UF_MAP = {
    11: "AC", 12: "AL", 13: "AM", 14: "AP", 15: "BA", 16: "CE", 17: "DF",
    21: "MA", 22: "SE", 23: "PA", 24: "PB", 25: "PE", 26: "PI", 27: "RN",
    28: "RO", 29: "RR", 31: "MG", 32: "ES", 33: "RJ", 35: "SP", 41: "PR",
    42: "SC", 43: "RS", 50: "MS", 51: "MT", 52: "GO", 53: "TO",
}


st.set_page_config(
    page_title="Painel de Dengue",
    page_icon="🦟",
    layout="wide",
)

DATA_DIR = "dados"


@st.cache_data(show_spinner="Carregando e processando os dados de dengue...")
def load_data() -> pd.DataFrame:
    """Carrega e consolida as notificações de dengue (2024-2026)."""
    cols = [
        "DT_NOTIFIC",
        "NU_ANO",
        "ID_MUNICIP",
        "SG_UF_NOT",
        "CLASSI_FIN",
        "EVOLUCAO",
        "HOSPITALIZ",
    ]

    df25 = pd.read_csv(
        f"{DATA_DIR}/DENGBR25.csv",
        sep=",",
        encoding="latin-1",
        low_memory=False,
        usecols=cols,
        dtype=str,
    )
    df25 = df25[df25["NU_ANO"].isin(["2024", "2025"])]

    df26 = pd.read_csv(
        f"{DATA_DIR}/DENGBR26.csv",
        sep=",",
        encoding="latin-1",
        low_memory=False,
        usecols=cols,
        dtype=str,
    )

    df = pd.concat([df25, df26], ignore_index=True)

    df["DT_NOTIFIC"] = pd.to_datetime(df["DT_NOTIFIC"], errors="coerce")
    df = df.dropna(subset=["DT_NOTIFIC"])
    df = df[df["DT_NOTIFIC"] <= pd.Timestamp.today().normalize()]

    df["ID_MUNICIP"] = df["ID_MUNICIP"].str.zfill(6)
    df["ano_mes"] = df["DT_NOTIFIC"].dt.to_period("M").astype(str)
    df["semana"] = df["DT_NOTIFIC"].dt.to_period("W").astype(str)

    df["obito"] = df["EVOLUCAO"].isin(["2", "3"]).astype(int)
    df["hospitalizado"] = (df["HOSPITALIZ"] == "1").astype(int)

    uf_map_str = {str(k): v for k, v in UF_MAP.items()}
    df["UF"] = (
        df["SG_UF_NOT"].astype(str).str.strip().map(uf_map_str).fillna(df["SG_UF_NOT"].astype(str))
    ).str.upper()

    return df


@st.cache_data(show_spinner="Carregando municípios...")
def load_municipios() -> pd.DataFrame:
    m = pd.read_csv(f"{DATA_DIR}/municipios.csv")
    m["codigo_ibge"] = m["codigo_ibge"].astype(int)
    m["id_municip"] = (m["codigo_ibge"] // 10).astype(str).str.zfill(6)
    return m[["id_municip", "nome", "latitude", "longitude", "codigo_uf"]]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega métricas por município para o clustering de criticidade."""
    g = df.groupby("ID_MUNICIP").agg(
        casos=("ID_MUNICIP", "size"),
        obitos=("obito", "sum"),
        hospitalizados=("hospitalizado", "sum"),
        UF=("UF", "first"),
    )
    g["taxa_hospitalizacao"] = np.where(g["casos"] > 0, g["hospitalizados"] / g["casos"], 0)
    g["taxa_obito"] = np.where(g["casos"] > 0, g["obitos"] / g["casos"], 0)
    g = g.reset_index().rename(columns={"ID_MUNICIP": "id_municip"})
    return g


def apply_criticality(features: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """Calcula score de criticidade e rótulos de cluster."""
    f = features.copy()
    f["log_casos"] = np.log1p(f["casos"])

    n_clusters = min(n_clusters, max(2, f.shape[0] - 1))

    scaler = StandardScaler()
    X = scaler.fit_transform(f[["log_casos", "taxa_hospitalizacao", "taxa_obito"]])

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    f["cluster"] = km.fit_predict(X)

    for col in ["casos", "taxa_hospitalizacao", "taxa_obito"]:
        f[f"norm_{col}"] = (f[col] - f[col].min()) / (f[col].max() - f[col].min() + 1e-9)

    f["score_criticidade"] = (
        0.5 * f["norm_casos"]
        + 0.3 * f["norm_taxa_obito"]
        + 0.2 * f["norm_taxa_hospitalizacao"]
    )
    f["score_criticidade"] = (f["score_criticidade"] * 100).round(1)

    # Mapeia cada cluster ao seu nível de criticidade pelo ranking da média do score.
    order = (
        f.groupby("cluster")["score_criticidade"].mean().sort_values(ascending=False).index.tolist()
    )
    tier_labels = ["Crítico", "Alto", "Médio", "Baixo"]
    cluster_to_tier = {c: tier_labels[i] if i < len(tier_labels) else "Baixo" for i, c in enumerate(order)}
    f["nivel"] = f["cluster"].map(cluster_to_tier)
    return f.sort_values("score_criticidade", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def csv_notificacoes(df: pd.DataFrame) -> bytes:
    """Gera o CSV das notificações filtradas (computado uma única vez)."""
    return df.to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False)
def csv_municipios(df: pd.DataFrame) -> bytes:
    """Gera o CSV de municípios/criticidade (computado uma única vez)."""
    colunas = [
        "nome", "UF", "id_municip", "casos", "obitos", "hospitalizados",
        "taxa_hospitalizacao", "taxa_obito", "score_criticidade", "nivel",
        "latitude", "longitude",
    ]
    colunas = [c for c in colunas if c in df.columns]
    return df[colunas].to_csv(index=False).encode("utf-8")


def forecast_arima(series: pd.Series, steps: int = 12):
    """Ajusta SARIMA sazonal (m=12) e projeta `steps` meses à frente."""
    from pmdarima import auto_arima

    y = series.values.astype(float)
    try:
        model = auto_arima(
            y,
            seasonal=True,
            m=12,
            d=1,
            D=1,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_p=3, max_q=3, max_P=2, max_Q=2,
        )
    except Exception:
        model = auto_arima(
            y, d=1, seasonal=False, stepwise=True, suppress_warnings=True, error_action="ignore"
        )
    fc = model.predict(n_periods=steps, return_conf_int=True, alpha=0.05)
    pred = np.maximum(fc[0], 0)
    lower = np.maximum(fc[1][:, 0], 0)
    upper = fc[1][:, 1]
    return pred, lower, upper, model


def main() -> None:
    st.title("🦟 Painel de Análise de Dengue")
    st.caption(
        "Fonte: SINAN (notificações de dengue, Brasil, 2024-2026). "
        "Cada registro representa um caso notificado de dengue."
    )

    df = load_data()
    municipios = load_municipios()

    ufs = sorted(df["UF"].dropna().unique().tolist())

    with st.sidebar:
        st.header("Filtros")
        uf_sel = st.multiselect("UF(s)", ufs, default=[])
        n_clusters = st.slider("Nº de clusters de criticidade", 3, 6, 4)
        horizonte = st.slider("Horizonte de previsão (meses)", 3, 12, 12)
        st.info(
            "Deixe 'UF(s)' vazio para considerar todo o Brasil. "
            "O último mês é parcial e não entra no treino do modelo."
        )

    if uf_sel:
        df_f = df[df["UF"].isin(uf_sel)].copy()
        escopo = " / ".join(uf_sel)
    else:
        df_f = df.copy()
        escopo = "Brasil (todas as UFs)"

    # Features de criticidade (uma única vez) — restritas a municípios
    # com casos acima do 1º quartil (remove 1 caso ou valores muito baixos).
    base_features = build_features(df_f)
    q1 = base_features["casos"].quantile(0.25)
    base_features = base_features[base_features["casos"] > q1]
    features = apply_criticality(base_features, n_clusters)
    features = features.merge(municipios, on="id_municip", how="left")
    features["nome"] = features["nome"].fillna(features["id_municip"])

    # Exportação dos dados que alimentam o dashboard
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📤 Exportar dados")
    csv_filt = csv_notificacoes(df_f)
    st.sidebar.download_button(
        "⬇️ Notificações filtradas (CSV)", csv_filt, "dengue_notificacoes_filtradas.csv", "text/csv"
    )
    csv_mun = csv_municipios(features)
    st.sidebar.download_button(
        "⬇️ Municípios e criticidade (CSV)", csv_mun, "dengue_municipios_criticalidade.csv", "text/csv"
    )
    st.sidebar.caption(
        "O CSV de notificações reflete o filtro de UF aplicado; "
        "o de municípios traz o score de criticidade usado no mapa e nas tabelas."
    )

    tab_prev, tab_crit, tab_acao = st.tabs(
        ["📈 Previsão (ARIMA)", "🗺️ Municípios Críticos", "🎯 Priorização e Ações"]
    )

    # ---------- Aba 1: Previsão ----------
    with tab_prev:
        st.subheader(f"Previsão de casos — {escopo}")
        monthly = df_f.groupby("ano_mes").size().sort_index()
        monthly.index = pd.PeriodIndex(monthly.index, freq="M").to_timestamp()

        if len(monthly) < 12:
            st.warning("Série temporal muito curta (mínimo de 12 meses) para o ARIMA com os filtros atuais.")
        else:
            # O mês mais recente costuma estar incompleto (dados ainda chegando);
            # se ele não cobre o mês inteiro, mantemos apenas para exibição e
            # treinamos o modelo com os meses completos anteriores.
            ultimo_dt = df_f["DT_NOTIFIC"].max()
            mes_atual = monthly.index[-1]
            mes_completo = (mes_atual + pd.offsets.MonthEnd(0)) <= ultimo_dt

            train = monthly if mes_completo else monthly.iloc[:-1]
            mes_parcial = None if mes_completo else monthly.tail(1)

            with st.spinner("Ajustando modelo SARIMA..."):
                try:
                    pred, lower, upper, model = forecast_arima(train, steps=horizonte)
                    future_idx = pd.date_range(
                        train.index[-1] + pd.offsets.MonthBegin(1), periods=horizonte, freq="MS"
                    )
                    fc_series = pd.Series(pred, index=future_idx)
                    ci_lower = pd.Series(lower, index=future_idx)
                    ci_upper = pd.Series(upper, index=future_idx)

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(x=train.index, y=train.values, name="Histórico (completo)", mode="lines")
                    )
                    if mes_parcial is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=[mes_parcial.index[0]], y=[mes_parcial.values[0]],
                                name="Mês atual (parcial)", mode="markers",
                                marker=dict(size=10, color="gray"),
                            )
                        )
                    fig.add_trace(
                        go.Scatter(
                            x=future_idx, y=fc_series.values, name="Previsão", mode="lines+markers",
                            line=dict(dash="dash", color="red"),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=pd.concat([pd.Series(future_idx), pd.Series(future_idx[::-1])]),
                            y=pd.concat([ci_upper, ci_lower[::-1]]),
                            fill="toself", fillcolor="rgba(255,0,0,0.1)",
                            line=dict(color="rgba(255,0,0,0)"), name="IC 95%",
                        )
                    )
                    fig.update_layout(
                        title=f"Casos mensais de dengue e projeção ({horizonte} meses) — SARIMA",
                        xaxis_title="Mês", yaxis_title="Casos notificados",
                        hovermode="x unified", height=450,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    proj_prox = int(round(fc_series.iloc[0]))
                    proj_3 = int(round(fc_series.iloc[:3].sum()))
                    proj_6 = int(round(fc_series.iloc[:6].sum()))
                    proj_total = int(round(fc_series.sum()))
                    ultimo_completo = int(train.iloc[-1])

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Último mês completo", f"{ultimo_completo:,}")
                    c2.metric("Projeção próximo mês", f"{proj_prox:,}")
                    c3.metric(f"Projeção {min(horizonte,6)} meses", f"{proj_6:,}")
                    c4.metric("Casos no período filtrado", f"{len(df_f):,}")

                    st.markdown(
                        f"**Resumo da projeção ({horizonte} meses):** "
                        f"próximo mês ≈ {proj_prox:,} · "
                        f"3 meses ≈ {proj_3:,} · "
                        f"6 meses ≈ {proj_6:,} · "
                        f"total ≈ {proj_total:,} casos."
                    )
                    st.caption(
                        f"Modelo: SARIMA {model.order}x{model.seasonal_order} "
                        f"(AIC={float(model.aic()):.0f}). "
                        "O mês atual parcial não entra no treino. IC 95% baseado no erro do modelo."
                    )
                except Exception as e:
                    st.error(f"Não foi possível ajustar o ARIMA: {e}")

        if len(monthly):
            monthly_df = monthly.reset_index()
            monthly_df.columns = ["Mês", "Casos"]
            fig_m = px.bar(monthly_df, x="Mês", y="Casos", title="Casos mensais (histórico)")
            fig_m.update_layout(height=350)
            st.plotly_chart(fig_m, use_container_width=True)

    # ---------- Aba 2: Municípios Críticos ----------
    with tab_crit:
        st.subheader(f"Municípios mais críticos — {escopo}")
        st.markdown(
            "Análise restrita a municípios com número de casos acima do 1º quartil "
            f"(Q1 = {int(q1)} casos), removendo registros com 1 caso ou muito poucos."
        )

        top_n = st.slider("Exibir top N municípios na tabela", 10, 50, 20)

        # Mapa do Brasil em destaque
        mapa_df = features.dropna(subset=["latitude"])
        if len(mapa_df):
            fig_map = px.scatter_geo(
                mapa_df, lat="latitude", lon="longitude",
                size="casos", color="nivel", hover_name="nome",
                scope="south america", size_max=35,
                title="Mapa de criticidade dos municípios (Brasil)",
                color_discrete_map={
                    "Crítico": "#b30000", "Alto": "#fc8d59",
                    "Médio": "#fee08b", "Baixo": "#91cf60",
                },
            )
            fig_map.update_layout(height=600, margin={"r": 0, "t": 40, "l": 0, "b": 0})
            st.plotly_chart(fig_map, use_container_width=True)

        # Top 15 por score (barras horizontais)
        bar = features.head(15).copy()
        bar["label"] = bar["nome"].str.slice(0, 28)
        fig_bar = px.bar(
            bar.sort_values("score_criticidade"),
            x="score_criticidade", y="label", color="nivel", orientation="h",
            title="Top 15 municípios por score de criticidade",
            labels={"score_criticidade": "Score", "label": "Município"},
            color_discrete_map={
                "Crítico": "#b30000", "Alto": "#fc8d59",
                "Médio": "#fee08b", "Baixo": "#91cf60",
            },
        )
        fig_bar.update_layout(height=520)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown(f"#### Ranking (top {top_n})")
        show = features.head(top_n)[
            ["nome", "UF", "id_municip", "casos", "obitos", "taxa_hospitalizacao", "score_criticidade", "nivel"]
        ].copy()
        show["taxa_hospitalizacao"] = (show["taxa_hospitalizacao"] * 100).round(1)
        show = show.rename(columns={
            "nome": "Município", "UF": "UF", "id_municip": "Código", "casos": "Casos",
            "obitos": "Óbitos", "taxa_hospitalizacao": "Tx. Hosp. (%)",
            "score_criticidade": "Score", "nivel": "Nível",
        })
        st.dataframe(show, use_container_width=True, hide_index=True)

    # ---------- Aba 3: Priorização e Ações ----------
    with tab_acao:
        st.subheader(f"Priorização e ações recomendadas — {escopo}")

        acoes = {
            "Crítico": (
                "🔴 Resposta imediata: acionar Força Tarefa Estadual, nebulização de bloqueio, "
                "ampliação de leitos e UTI, monitoramento diário, e investigação de óbitos."
            ),
            "Alto": (
                "🟠 Ação prioritária: mutirão de eliminação de criadouros, reforço de vigilância "
                "epidemiológica, capacitação de unidades para sinais de alarme e busca ativa."
            ),
            "Médio": (
                "🟡 Manter ações de rotina: educação em saúde, visitas domiciliares e monitoramento "
                "dos indicadores semanais."
            ),
            "Baixo": (
                "🟢 Monitoramento passivo: manter vigilância e preparar plano de contingência."
            ),
        }

        resumo = features.groupby("nivel").agg(
            municipios=("id_municip", "size"),
            casos=("casos", "sum"),
            obitos=("obitos", "sum"),
        ).reindex(["Crítico", "Alto", "Médio", "Baixo"]).dropna(how="all")

        st.markdown("#### Resumo por nível de criticidade")
        st.dataframe(resumo, use_container_width=True)

        st.markdown("#### Ranking de priorização (municípios)")
        pri = features[["nome", "UF", "casos", "obitos", "taxa_hospitalizacao", "score_criticidade", "nivel"]].copy()
        pri["taxa_hospitalizacao"] = (pri["taxa_hospitalizacao"] * 100).round(1)
        pri = pri.rename(columns={
            "nome": "Município", "UF": "UF", "casos": "Casos", "obitos": "Óbitos",
            "taxa_hospitalizacao": "Tx. Hosp. (%)", "score_criticidade": "Score", "nivel": "Nível",
        })
        st.dataframe(pri, use_container_width=True, hide_index=True, height=400)

        st.markdown("#### Ações recomendadas por nível")
        for nivel in ["Crítico", "Alto", "Médio", "Baixo"]:
            if nivel in resumo.index:
                st.markdown(f"**{nivel}** ({int(resumo.loc[nivel, 'municipios'])} municípios): {acoes[nivel]}")

        csv = pri.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Baixar ranking de priorização (CSV)", csv, "priorizacao_dengue.csv", "text/csv"
        )


if __name__ == "__main__":
    main()
