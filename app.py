import datetime as dt
import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import json

from dotenv import load_dotenv
from openai import OpenAI

# =============================
# Carga del .env y configuración de OpenAI
# =============================

# Ruta absoluta al .env en la misma carpeta que app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=dotenv_path)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "No se encontró la variable OPENAI_API_KEY.\n\n"
        "Verifica que:\n"
        "- Exista un archivo .env en la MISMA carpeta que app.py\n"
        "- Contenga la línea: OPENAI_API_KEY=tu_key_aquí\n"
        "- Estás ejecutando `streamlit run app.py` desde esta carpeta."
    )
    st.stop()

client = OpenAI(api_key=api_key)

# =============================
# Configuración básica de la app
# =============================
st.set_page_config(
    page_title="CAPM y Técnicos",
    layout="wide"
)
st.markdown("""
<style>

div[role="tablist"] > button {
    background-color: #1e293b !important;   /* gris azulado oscuro */
    color: #e2e8f0 !important;              /* gris claro */
    border-radius: 8px !important;
    margin-right: 6px !important;
    padding: 8px 14px !important;
    border: 1px solid #334155 !important;
    font-weight: 500 !important;
}

div[role="tablist"] > button[aria-selected="true"] {
    background-color: #334155 !important;
    color: #f8fafc !important;
    border: 1px solid #38bdf8 !important;   /* acento azul (igual al theme) */
}

div[role="tabpanel"] {
    padding-top: 1rem !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.fin-table {
    border-collapse: collapse;
    width: 100%;
    font-size: 13px;
}
.fin-table thead tr {
    background-color: #020617;
    color: #e5e7eb;
}
.fin-table th, .fin-table td {
    border: 1px solid #1f2937;
    padding: 6px 10px;
    text-align: right;
}
.fin-table th:first-child,
.fin-table td:first-child {
    text-align: left;
}
.fin-table tbody tr:nth-child(even) {
    background-color: #020617;
}
.fin-table tbody tr:nth-child(odd) {
    background-color: #020617dd;
}
.fin-table tbody tr.highlight-row {
    background-color: #1d4ed833 !important;
}
</style>
""", unsafe_allow_html=True)



# =============================
# Landing: título y botón de inicio
# =============================
if "started" not in st.session_state:
    st.session_state["started"] = False

st.title("Guía rápida para invertir en los mercados")
st.caption(
    "Dashboard financiero interactivo que integra análisis CAPM 📉, evaluación técnica de precios 📈, construcción de la frontera eficiente bajo el marco de Markowitz 🧮 y un módulo de asesoría asistida por IA 🤖. Una plataforma diseñada para ofrecer una lectura integral del mercado, apoyar la toma de decisiones y fortalecer tu enfoque estratégico como inversionista."
)

if not st.session_state["started"]:
    st.markdown(
        """
        ### ¿Listo para triunfar de los mercados?

        Esta herramienta te permite:

        - Analizar un ticker vs su benchmark con CAPM y beta rolling.  
        - Revisar momentum, tendencia y niveles clave con técnicos (RSI, MACD, SMA, Fibonacci, ADX).  
        - Construir una frontera eficiente con varios activos y ajustar según tu aversión al riesgo.  
        - Pedirle a un agente de IA que te resuma todo en lenguaje formal o en un tono más amigable.

        Cuando estés listo, dale al botón.
        """
    )

    if st.button("Comenzar", type="primary", use_container_width=True):
        st.session_state["started"] = True
        st.rerun()

    # No renderizamos nada más hasta que el usuario apriete el botón
    st.stop()


# =============================
# Sidebar: parámetros
# =============================
st.sidebar.title("Panel de control")

# --------- Bloque GLOBAL ---------
st.sidebar.markdown("### Parámetros globales")

ticker = st.sidebar.text_input("Ticker principal", "AAPL", key="ticker_input").upper()
benchmark = st.sidebar.text_input("Benchmark (mercado)", "SPY", key="benchmark_input")
st.sidebar.markdown(
    """
    <div style="font-size:11px; color:#94a3b8; margin-top:-6px;">
        Tickers de referencia más usados:  
        📌 <strong>SPY</strong> (S&P 500 ETF)  
        📌 <strong>QQQ</strong> (Nasdaq 100 ETF)  
        📌 <strong>^GSPC</strong> (S&P 500 Index)  
        📌 <strong>^NDX</strong> (Nasdaq 100 Index)  
        📌 <strong>^DJI</strong> (Dow Jones Index)
    </div>
    """,
    unsafe_allow_html=True,
)
    
years = st.sidebar.slider("Años de historia", 1, 10, 3, key="years_slider")

show_benchmark = st.sidebar.checkbox(
    "Mostrar benchmark en Overview",
    True,
    key="show_bench_checkbox"
)

st.sidebar.markdown("---")

# --------- Bloque 2: Técnicos ---------
with st.sidebar.expander("Indicadores técnicos"):
    st.sidebar.caption("Configuración para la pestaña Technicals.")

    show_ma   = st.checkbox("SMA 20/50/200", True, key="show_ma_checkbox")
    show_rsi  = st.checkbox("RSI", True, key="show_rsi_checkbox")
    show_macd = st.checkbox("MACD", True, key="show_macd_checkbox")
    show_adx  = st.checkbox("ADX", False, key="show_adx_checkbox")
    show_fibo = st.checkbox("Niveles Fibonacci", False, key="show_fibo_checkbox")

    st.markdown("**Ajustes finos**")

    sma_periods_str = st.text_input(
        "SMA (periodos, separados por coma)",
        "20,50,200",
        key="sma_periods_input"
    )
    try:
        sma_periods = [int(x.strip()) for x in sma_periods_str.split(",")]
    except Exception:
        sma_periods = [20, 50, 200]

    rsi_period = st.number_input(
        "RSI – periodo",
        min_value=5,
        max_value=50,
        value=14,
        key="rsi_period_input"
    )

    macd_fast = st.number_input(
        "MACD – EMA rápida",
        min_value=5,
        max_value=50,
        value=12,
        key="macd_fast_input"
    )
    macd_slow = st.number_input(
        "MACD – EMA lenta",
        min_value=10,
        max_value=100,
        value=26,
        key="macd_slow_input"
    )
    macd_signal = st.number_input(
        "MACD – señal",
        min_value=5,
        max_value=30,
        value=9,
        key="macd_signal_input"
    )

    adx_period = st.number_input(
        "ADX – periodo",
        min_value=5,
        max_value=50,
        value=14,
        key="adx_period_input"
    )

    fibo_lookback = st.number_input(
        "Fibonacci – ventana (días)",
        min_value=30,
        max_value=300,
        value=120,
        key="fibo_lookback_input"
    )

st.sidebar.markdown("---")

# --------- Bloque 3: CAPM ---------
with st.sidebar.expander("CAPM"):
    st.sidebar.caption("Parámetros para riesgo y retorno esperado.")

    rf_annual_pct = st.number_input(
        "Tasa libre de riesgo anual (%)",
        min_value=-5.0,
        max_value=15.0,
        value=4.0,
        step=0.25,
        key="rf_annual_input"
    )

    beta_window = st.number_input(
        "Ventana beta rolling (días)",
        min_value=60,
        max_value=500,
        value=252,
        step=21,
        key="beta_window_input"
    )

st.sidebar.markdown("---")

# --------- Bloque 4: Frontera & Portafolio ---------
with st.sidebar.expander("Frontera eficiente y portafolio"):
    st.sidebar.caption("Activos adicionales y aversión al riesgo.")

    extra1 = st.text_input("Ticker extra 1", "", key="extra1_input").upper()
    extra2 = st.text_input("Ticker extra 2", "", key="extra2_input").upper()
    extra3 = st.text_input("Ticker extra 3", "", key="extra3_input").upper()

    risk_aversion = st.slider(
        "Aversión al riesgo A",
        min_value=1.0,
        max_value=10.0,
        value=4.0,
        step=0.5,
        key="risk_aversion_slider",
        help="Valores bajos = inversionista agresivo; valores altos = conservador."
    )

st.sidebar.markdown("---")

# --------- Bloque 5: AI Advisor ---------
with st.sidebar.expander("Agente de IA"):
    st.sidebar.caption("Define el tono del agente de IA.")

    paps_mode = st.checkbox(
        "Paps de los mercados",
        value=False,
        key="paps_mode_checkbox",
        help=(
            "Si está activado, el AI Advisor hablará en tono informal con toques de comedia. "
            "Si está desactivado, el tono será formal y profesional."
        ),
    )

# =============================
# Tabs principales
# =============================
tab_overview, tab_tech, tab_capm, tab_frontier, tab_ai = st.tabs(
    [
        "📊 Resumen",
        "📈 Análisis Técnico",
        "📉 CAPM",
        "🧮 Frontera Eficiente",
        "🤖 Agente de IA"
    ]
)

# =============================
# Barra de contexto del modelo
# =============================
with st.container():
    st.markdown(
        """
        <div style="
            background-color:#0b1320;
            border-radius:12px;
            padding:14px 18px;
            margin-bottom: 14px;
            border:1px solid #1f2937;
        ">
            <div style="display:flex; flex-wrap:wrap; gap:18px; align-items:center;">
                <div style="font-weight:600; font-size:16px; color:#e5e7eb;">
                    📊 Contexto actual del modelo
                </div>
                <div style="font-size:13px; color:#9ca3af;">
                    Ticker: <span style="color:#f9fafb; font-weight:600;">{ticker}</span>
                </div>
                <div style="font-size:13px; color:#9ca3af;">
                    Benchmark: <span style="color:#f9fafb; font-weight:600;">{benchmark}</span>
                </div>
                <div style="font-size:13px; color:#9ca3af;">
                    Historial: <span style="color:#f9fafb; font-weight:600;">{years} años</span>
                </div>
                <div style="font-size:13px; color:#9ca3af;">
                    Rf anual: <span style="color:#f9fafb; font-weight:600;">{rf_annual_pct:.2f}%</span>
                </div>
                <div style="font-size:13px; color:#9ca3af;">
                    Aversión A: <span style="color:#f9fafb; font-weight:600;">{risk_aversion:.1f}</span>
                </div>
            </div>
        </div>
        """.format(
            ticker=ticker,
            benchmark=benchmark,
            years=years,
            rf_annual_pct=rf_annual_pct,
            risk_aversion=risk_aversion,
        ),
        unsafe_allow_html=True,
    )


# =============================
# Funciones helper sencillas
# =============================
@st.cache_data(show_spinner=False)
def download_prices(symbol: str, years_back: int) -> pd.DataFrame:
    end = dt.date.today()
    start = end - dt.timedelta(days=int(365.25 * years_back) + 10)

    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",   # importante para que venga por columnas
    )

    # Si viene como MultiIndex tipo (Open, AAPL) lo aplanamos
    if isinstance(df.columns, pd.MultiIndex):
        # Nos quedamos con el primer nivel: Open, High, Low, Close...
        df.columns = df.columns.get_level_values(0)

    return df

@st.cache_data(show_spinner=False)
def get_ticker_profile(symbol: str) -> dict:
    """
    Devuelve un dict con algunos datos clave del ticker usando yfinance.Ticker.
    Si no se puede obtener, regresa dict vacío.
    """
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
    except Exception:
        return {}

    profile = {
        "name": info.get("longName") or info.get("shortName") or symbol,
        "exchange": info.get("fullExchangeName") or info.get("exchange"),
        "currency": info.get("currency"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "beta_yahoo": info.get("beta"),
    }
    return profile

def price_chart(close: pd.Series, overlays: dict | None = None,
                fib_levels: dict | None = None, title: str = "Precio"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close.values, name="Close"))

    if overlays:
        for name, series in overlays.items():
            fig.add_trace(go.Scatter(x=series.index, y=series.values, name=name))

    if fib_levels:
        for name, level in fib_levels.items():
            fig.add_hline(
                y=level,
                line_dash="dot",
                annotation_text=f"Fib {name}",
                annotation_position="top left"
            )

    fig.update_layout(
        title=title,
        height=350,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

def candle_chart_basic(ohlc: pd.DataFrame, title: str):
    """
    Recibe un DataFrame con columnas Open, High, Low, Close ya limpias.
    """
    if ohlc.empty:
        return None

    # Aseguramos tipo float
    ohlc = ohlc.astype(float)

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=ohlc.index,
                open=ohlc["Open"],
                high=ohlc["High"],
                low=ohlc["Low"],
                close=ohlc["Close"],
                name=title,
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title="Precio",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
    )
    return fig

def compute_sma(close: pd.Series, windows=(20, 50, 200)) -> dict:
    out = {}
    for w in windows:
        out[f"SMA{w}"] = close.rolling(window=w).mean()
    return out


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    high_shift = high.shift(1)
    low_shift = low.shift(1)
    close_shift = close.shift(1)

    plus_dm = (high - high_shift).where((high - high_shift) > (low_shift - low), 0.0)
    plus_dm = plus_dm.where(plus_dm > 0, 0.0)

    minus_dm = (low_shift - low).where((low_shift - low) > (high - high_shift), 0.0)
    minus_dm = minus_dm.where(minus_dm > 0, 0.0)

    tr1 = high - low
    tr2 = (high - close_shift).abs()
    tr3 = (low - close_shift).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).sum()

    plus_di = 100 * (plus_dm.rolling(window=period).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).sum() / atr)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).abs() * 100
    adx = dx.rolling(window=period).mean()
    return adx, plus_di, minus_di


def compute_fibonacci(close: pd.Series, lookback: int = 120) -> dict:
    recent = close.dropna().tail(lookback)
    if recent.empty:
        return {}

    hi = recent.max()
    lo = recent.min()
    diff = hi - lo

    levels = {
        "0.0": lo,
        "0.236": hi - 0.236 * diff,
        "0.382": hi - 0.382 * diff,
        "0.5": (hi + lo) / 2,
        "0.618": hi - 0.618 * diff,
        "0.786": hi - 0.786 * diff,
        "1.0": hi,
    }
    return levels

def compute_capm_metrics(
    close_asset: pd.Series,
    close_bench: pd.Series,
    rf_annual: float = 0.04,
) -> dict:
    """
    Calcula métricas CAPM básicas: beta, alpha, retornos anualizados, Sharpe, etc.
    rf_annual en formato decimal (0.04 = 4%).
    """
    df = pd.concat(
        [close_asset.rename("asset"), close_bench.rename("bench")],
        axis=1,
        join="inner"
    ).dropna()

    r = df.pct_change().dropna()
    if r.empty:
        raise ValueError("No hay suficientes retornos para CAPM.")

    r_a = r["asset"]
    r_m = r["bench"]

    # Tasa libre de riesgo diaria
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1

    # Excess returns
    ex_a = r_a - rf_daily
    ex_m = r_m - rf_daily

    # Beta y alpha (diarios)
    cov_am = ex_a.cov(ex_m)
    var_m = ex_m.var()
    beta = cov_am / var_m if var_m != 0 else np.nan

    alpha_daily = ex_a.mean() - beta * ex_m.mean()
    # Aproximación lineal para anualizar alpha
    alpha_annual = alpha_daily * 252

    # Correlación y R2
    corr = ex_a.corr(ex_m)
    r2 = corr**2

    # Retornos anualizados
    n = len(r_a)
    asset_ann = (1 + r_a).prod() ** (252 / n) - 1
    market_ann = (1 + r_m).prod() ** (252 / n) - 1

    # Volatilidad anualizada del activo
    asset_ann_vol = r_a.std() * np.sqrt(252)

    # Expected return CAPM
    exp_capm = rf_annual + beta * (market_ann - rf_annual) if not np.isnan(beta) else np.nan

    # Sharpe usando retorno realizado del activo
    sharpe = (
        (asset_ann - rf_annual) / asset_ann_vol
        if asset_ann_vol not in (0, np.nan) and not np.isnan(asset_ann_vol)
        else np.nan
    )

    return {
        "returns_df": r,
        "excess_asset": ex_a,
        "excess_market": ex_m,
        "beta": beta,
        "alpha_annual": alpha_annual,
        "r2": r2,
        "asset_ann": asset_ann,
        "market_ann": market_ann,
        "asset_ann_vol": asset_ann_vol,
        "exp_capm": exp_capm,
        "sharpe": sharpe,
    }


def compute_rolling_beta(
    returns_df: pd.DataFrame,
    window: int = 252,
) -> pd.Series:
    """
    Beta rolling usando var/cov sobre retornos simples (no excesos).
    returns_df con columnas ['asset','bench'].
    """
    roll_cov = returns_df["asset"].rolling(window).cov(returns_df["bench"])
    roll_var = returns_df["bench"].rolling(window).var()
    beta_roll = roll_cov / roll_var
    beta_roll.name = f"Beta_{window}d"
    return beta_roll

def build_ai_summary(
    ticker: str,
    benchmark: str,
    rf_annual: float,
    capm_metrics: dict,
    markowitz_result: dict,
) -> dict:
    """
    Prepara un diccionario compacto con los números clave que usará el prompt.
    Todo lo que va aquí son números y etiquetas simples.
    """
    return {
        "ticker": ticker,
        "benchmark": benchmark,
        "risk_free_annual": rf_annual,
        "capm": {
            "beta": capm_metrics.get("beta"),
            "alpha_annual": capm_metrics.get("alpha_annual"),
            "asset_ann_return": capm_metrics.get("asset_ann"),
            "market_ann_return": capm_metrics.get("market_ann"),
            "asset_ann_vol": capm_metrics.get("asset_ann_vol"),
            "expected_capm_return": capm_metrics.get("exp_capm"),
            "sharpe_realized": capm_metrics.get("sharpe"),
            "r2": capm_metrics.get("r2"),
        },
        "markowitz": {
            "portfolio_assets": markowitz_result.get("assets", []),
            "weights_optimal": markowitz_result.get("weights", []),
            "portfolio_return_optimal": markowitz_result.get("ret_star"),
            "portfolio_vol_optimal": markowitz_result.get("vol_star"),
            "portfolio_sharpe_optimal": markowitz_result.get("sharpe_star"),
            "risk_aversion_A": markowitz_result.get("A"),
        },
    }

def sanitize_for_json(obj):
    """Convierte todo en tipos nativos de Python."""
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    # numpy scalars
    try:
        import numpy as np
        if isinstance(obj, (np.float64, np.float32, np.int64, np.int32)):
            return obj.item()
    except:
        pass
    # fallback
    return str(obj)


def call_ai_advisor(summary: dict, paps_mode: bool) -> str:
    """
    Llama al modelo de OpenAI usando la nueva API.
    """

    # Convertimos todo para que json.dumps no truene
    summary_clean = sanitize_for_json(summary)
    summary_json = json.dumps(summary_clean, indent=2)

    if paps_mode:
        system_msg = (
            "Hablas como un analista financiero 'fresa/whitexican' mexicano, "
            "usando expresiones como 'pa', 'bro', 'muy top', 'heavy', vibrar alto, animo, lowkey etc. "
            "El análisis sigue siendo serio y cuantitativo, pero el tono es relajado y cómico sin ser vulgar."
        )
    else:
        system_msg = (
            "Eres un analista cuantitativo profesional. "
            "Analiza la situación del mercado, el ticker, CAPM y el portafolio Markowitz con precisión. "
            "No inventes datos. Sé claro, formal y estructurado."
        )

    user_msg = (
        "Con base en los datos siguientes, entrega un análisis en secciones:\n"
        "1) Resumen ejecutivo.\n"
        "2) Lectura del mercado.\n"
        "3) Lectura del ticker.\n"
        "4) Lectura del portafolio óptimo.\n"
        "5) Recomendaciones tácticas y estratégicas.\n\n"
        "DATOS:\n"
        f"```json\n{summary_json}\n```"
    )

    # ===== NUEVA API DE OPENAI =====
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_output_tokens=900,
        temperature=0.85 if paps_mode else 0.55,
    )

    # El texto viene aquí
    return response.output_text


# =============================
# Lógica Bloque 1: solo Overview
# =============================
with tab_overview:
    st.subheader(f"Resumen de {ticker}")
    st.caption("Vista rápida de precio, riesgo y comparación con el benchmark.")


    with st.spinner("Descargando datos desde yfinance..."):
        data_ticker = download_prices(ticker, years)
        data_bench = download_prices(benchmark, years) if show_benchmark else None

    if data_ticker.empty:
        st.error("No se encontraron datos para ese ticker. Prueba con otro símbolo (ej. MSFT, TSLA, NVDA).")
    else:
        # --------- OHLC directo ----------
        try:
            ohlc = data_ticker[["Open", "High", "Low", "Close"]].dropna()
        except KeyError as e:
            st.error(f"El DataFrame no contiene columna {e}. Columnas actuales: {list(data_ticker.columns)}")
            st.stop()

        close = ohlc["Close"]

        # ========= MÉTRICAS BÁSICAS + RIESGO/RETORNO =========
        last_price = float(close.iloc[-1])
        first_date = close.index[0].date()
        last_date = close.index[-1].date()

        # Retornos diarios
        r = close.pct_change().dropna()
        if len(r) > 0:
            last_ret_1d = r.iloc[-1]
            cum_ret = close.iloc[-1] / close.iloc[0] - 1
            n = len(r)
            ann_ret = (1 + cum_ret) ** (252 / n) - 1
            ann_vol = r.std() * np.sqrt(252)
        else:
            last_ret_1d = cum_ret = ann_ret = ann_vol = np.nan

        # Fila 1: precio y fechas
        col1, col2, col3 = st.columns(3)
        col1.metric("Último precio", f"{last_price:,.2f}")
        col2.metric("Fecha inicial", str(first_date))
        col3.metric("Fecha final", str(last_date))

        # Fila 2: performance y riesgo
        col4, col5, col6 = st.columns(3)
        col4.metric("Cambio 1D", f"{last_ret_1d*100:0.2f}%" if not np.isnan(last_ret_1d) else "NA")
        col5.metric("Rendimiento periodo", f"{cum_ret*100:0.2f}%" if not np.isnan(cum_ret) else "NA")
        col6.metric(
            "Retorno/Vol anualizados",
            f"{ann_ret*100:0.2f}% / {ann_vol*100:0.2f}%" if not np.isnan(ann_ret) else "NA"
        )

        # ========= PERFIL DEL ACTIVO =========
        profile = get_ticker_profile(ticker)
        with st.expander("Ficha del activo (yfinance)", expanded=False):
            if not profile:
                st.write("No se pudo obtener información del activo.")
            else:
                mc = profile.get("market_cap")
                mc_str = f"{mc:,.0f}" if mc is not None else "NA"
                st.markdown(f"""
                **Nombre:** {profile.get('name', 'NA')}  
                **Exchange:** {profile.get('exchange', 'NA')}  
                **Moneda:** {profile.get('currency', 'NA')}  
                **Sector:** {profile.get('sector', 'NA')}  
                **Industria:** {profile.get('industry', 'NA')}  
                **Market cap:** {mc_str}  
                **Beta (Yahoo):** {profile.get('beta_yahoo', 'NA')}
                """)

        # ========= GRÁFICO 1: VELAS =========
        st.markdown("### Velas del ticker")
        fig_candles = candle_chart_basic(ohlc, f"{ticker} OHLC")
        if fig_candles is None:
            st.warning("No se pudieron generar velas (DataFrame OHLC vacío).")
        else:
            st.plotly_chart(fig_candles, use_container_width=True)

        # ========= GRÁFICO 2: RELATIVO VS BENCHMARK =========
        if show_benchmark and data_bench is not None and not data_bench.empty:
            st.markdown("### Rendimiento relativo vs benchmark (normalizado a 1)")

            try:
                bench_close = data_bench["Close"].dropna()
            except KeyError:
                bench_close = None

            if bench_close is not None and not bench_close.empty:
                joined = pd.concat(
                    [close, bench_close],
                    axis=1,
                    join="inner"
                )
                joined.columns = [ticker, benchmark]
                norm = joined / joined.iloc[0]  # ojo, si aquí falla, corrige a joined.iloc[0]

                fig_rel = go.Figure()
                fig_rel.add_trace(
                    go.Scatter(
                        x=norm.index,
                        y=norm[ticker],
                        name=f"{ticker} (norm)"
                    )
                )
                fig_rel.add_trace(
                    go.Scatter(
                        x=norm.index,
                        y=norm[benchmark],
                        name=f"{benchmark} (norm)"
                    )
                )
                fig_rel.update_layout(
                    title=f"{ticker} vs {benchmark} (normalizados a 1)",
                    height=320,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_rel, use_container_width=True)
            else:
                st.warning("No se pudieron obtener precios de cierre válidos para el benchmark.")

        st.caption(
            "Overview: métrica básica de precio, rendimiento y riesgo del periodo, "
            "velas del ticker y, opcionalmente, comparación normalizada con el benchmark. "
            "En los siguientes bloques añadimos indicadores técnicos y CAPM."
        )

# =============================
# Tab: Technicals
# =============================
with tab_tech:
    st.subheader(f"Indicadores técnicos de {ticker}")
    st.caption("Momentum, tendencia y niveles clave sobre el precio del ticker.")


    with st.spinner("Calculando indicadores..."):
        data_ticker_tech = download_prices(ticker, years)

    if data_ticker_tech.empty:
        st.error("No se encontraron datos para ese ticker.")
    else:
        try:
            ohlc_t = data_ticker_tech[["Open", "High", "Low", "Close"]].dropna()
        except KeyError as e:
            st.error(f"El DataFrame no contiene columna {e}. Columnas actuales: {list(data_ticker_tech.columns)}")
            st.stop()

        close_t = ohlc_t["Close"]
        high_t = ohlc_t["High"]
        low_t  = ohlc_t["Low"]

        # ---------- Precio + medias y Fibonacci ----------
        overlays = {}
        if show_ma:
            sma_dict = compute_sma(close_t, windows=tuple(sma_periods))
            overlays.update(sma_dict)

        fib_levels = compute_fibonacci(close_t, lookback=fibo_lookback)

        fig_price = price_chart(close_t, overlays=overlays if overlays else None,
                        fib_levels=fib_levels, title=f"{ticker} Close + SMA/Fib")
        st.plotly_chart(fig_price, use_container_width=True)

        with st.expander("Info del indicador: SMA y Fibonacci"):
            st.markdown("""
        **SMA (Simple Moving Average):**  
        Suaviza el precio y revela la tendencia dominante. Cruces entre SMA cortas y largas (por ejemplo, SMA20 vs SMA50/200) suelen señalar cambios estructurales en la tendencia.

        **Fibonacci:**  
        Toma el último swing relevante (máximo–mínimo) y proyecta niveles de retroceso comunes (0.382, 0.5, 0.618, 0.786).  
        Suelen actuar como zonas probables de soporte/resistencia donde el precio tiende a frenar, rebotar o consolidar.
        """)

        # ---------- RSI ----------
        if show_rsi:
            rsi = compute_rsi(close_t, period=rsi_period)
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=rsi.index, y=rsi.values, name="RSI(14)"))
            fig_rsi.add_hline(y=70, line_dash="dash")
            fig_rsi.add_hline(y=30, line_dash="dash")
            fig_rsi.update_layout(
                title="RSI (14)",
                height=250,
                margin=dict(l=10, r=10, t=40, b=10),
                yaxis_range=[0, 100]
            )
            st.plotly_chart(fig_rsi, use_container_width=True)

            with st.expander("Info del indicador: RSI"):
                st.markdown(f"""
                **RSI (Relative Strength Index, periodo {rsi_period}):**  
                Oscilador de momentum que mide la intensidad de los movimientos recientes.
        
                - Valores **> 70** suelen asociarse a sobrecompra (riesgo de corrección).  
                - Valores **< 30** suelen asociarse a sobreventa (posible rebote).  
                - Las **divergencias** entre RSI y precio (precio hace nuevos máximos pero RSI no) suelen anticipar giros de tendencia.
                """)

            # ---------- MACD ----------
            if show_macd:
                # Cálculo MACD con parámetros dinámicos del sidebar
                macd_line, signal_line, hist = compute_macd(
                    close_t,
                    fast=macd_fast,
                    slow=macd_slow,
                    signal=macd_signal
                )

                # Gráfica MACD
                fig_macd = go.Figure()

                # Histograma
                fig_macd.add_trace(
                    go.Bar(
                        x=hist.index,
                        y=hist.values,
                        name="Histograma",
                        marker_color="gray"
                    )
                )

                # Línea MACD
                fig_macd.add_trace(
                    go.Scatter(
                        x=macd_line.index,
                        y=macd_line.values,
                        name="MACD",
                        line=dict(color="#00b3ff", width=2)
                    )
                )

                # Línea de señal
                fig_macd.add_trace(
                    go.Scatter(
                        x=signal_line.index,
                        y=signal_line.values,
                        name="Señal",
                        line=dict(color="#ff6b6b", width=2)
                    )
                )

                # Layout dinámico según parámetros
                fig_macd.update_layout(
                    title=f"MACD ({macd_fast}, {macd_slow}, {macd_signal})",
                    height=280,
                    margin=dict(l=10, r=10, t=40, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig_macd, use_container_width=True)

                # Explicación debajo en barra expandible
                with st.expander("Info del indicador: MACD"):
                    st.markdown(f"""
            **MACD (Moving Average Convergence Divergence)**  
            Parámetros usados: **EMA rápida = {macd_fast}**, **EMA lenta = {macd_slow}**, **Señal = {macd_signal}**.

            **¿Qué mide?**  
            El MACD captura la diferencia entre las medias móviles exponenciales (EMA) rápida y lenta.  
            El histograma muestra la aceleración del momentum.

            **Interpretación clave:**  
            - **Cruce MACD → Señal al alza** → incremento de momentum alcista.  
            - **Cruce MACD → Señal a la baja** → incremento de momentum bajista.  
            - **Histograma creciente** → aceleración de la tendencia.  
            - **Histograma decreciente** → pérdida de impulso.  
            - MACD **> 0** → tendencia subyacente alcista.  
            - MACD **< 0** → tendencia subyacente bajista.
                    """)


            # ---------- ADX ----------
                if show_adx:
                    adx, plus_di, minus_di = compute_adx(
                        high_t, low_t, close_t, period=adx_period
                    )

                    fig_adx = go.Figure()
                    fig_adx.add_trace(go.Scatter(x=adx.index, y=adx.values, name="ADX"))
                    fig_adx.add_trace(go.Scatter(x=plus_di.index, y=plus_di.values, name="+DI"))
                    fig_adx.add_trace(go.Scatter(x=minus_di.index, y=minus_di.values, name="-DI"))

                    fig_adx.update_layout(
                        title=f"ADX ({adx_period})",
                        height=280,
                        margin=dict(l=10, r=10, t=40, b=10)
                    )

                    st.plotly_chart(fig_adx, use_container_width=True)

                    with st.expander("Info del indicador: ADX"):
                        st.markdown(f"""
                **ADX (Average Directional Index, periodo {adx_period}):**  
                Mide la **fuerza** de la tendencia, independientemente de si es alcista o bajista.

                - **ADX < 20** → mercado en rango / sin tendencia clara.  
                - **20–40** → tendencia razonablemente sólida.  
                - **> 40** → tendencia fuerte.  

                Las líneas **+DI** y **−DI** indican la dirección dominante:  
                - +DI > –DI → tendencia alcista predominante.  
                - –DI > +DI → tendencia bajista predominante.
                """)
    
# =============================
# CAPM Tab
# =============================

with tab_capm:
    st.subheader(f"CAPM y métricas de riesgo para {ticker}")
    st.caption("Beta, alpha, R² y beta rolling del activo frente al benchmark.")


    rf_annual = rf_annual_pct / 100.0

    with st.spinner("Calculando métricas CAPM..."):
        data_ticker_capm = download_prices(ticker, years)
        data_bench_capm = download_prices(benchmark, years)

    if data_ticker_capm.empty or data_bench_capm.empty:
        st.error("No se encontraron datos suficientes para el ticker o el benchmark.")
    else:
        try:
            close_a = data_ticker_capm["Close"].dropna()
            close_m = data_bench_capm["Close"].dropna()
        except KeyError as e:
            st.error(f"Faltan columnas de Close en los datos: {e}")
            st.stop()

        # Alineamos
        close_a, close_m = close_a.align(close_m, join="inner")

        try:
            metrics = compute_capm_metrics(close_a, close_m, rf_annual=rf_annual)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        r = metrics["returns_df"]
        ex_a = metrics["excess_asset"]
        ex_m = metrics["excess_market"]

        beta = metrics["beta"]
        alpha_annual = metrics["alpha_annual"]
        r2 = metrics["r2"]
        asset_ann = metrics["asset_ann"]
        market_ann = metrics["market_ann"]
        asset_ann_vol = metrics["asset_ann_vol"]
        exp_capm = metrics["exp_capm"]
        sharpe = metrics["sharpe"]

        # --------- Métricas clave CAPM ----------
        col1, col2, col3 = st.columns(3)
        col1.metric("Beta (CAPM)", f"{beta:0.3f}")
        col2.metric("Alpha anual (exceso)", f"{alpha_annual*100:0.2f}%")
        col3.metric("R² modelo", f"{r2:0.3f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Retorno anual activo", f"{asset_ann*100:0.2f}%")
        col5.metric("Retorno anual mercado", f"{market_ann*100:0.2f}%")
        col6.metric("Vol anual activo", f"{asset_ann_vol*100:0.2f}%")

        col7, col8 = st.columns(2)
        col7.metric("Retorno esperado (CAPM)", f"{exp_capm*100:0.2f}%")
        col8.metric("Sharpe (realizado)", f"{sharpe:0.2f}")

        # ========= Scatter CAPM: exceso activo vs exceso mercado =========
        st.markdown("### Relación CAPM: exceso de retorno del activo vs mercado")

        # Recta de regresión: ex_a = alpha + beta * ex_m
        x = ex_m
        y = ex_a
        reg_line = beta * x + y.mean() - beta * x.mean()  # usando media para intercepto

        fig_scatter = go.Figure()
        fig_scatter.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Observaciones",
                opacity=0.4
            )
        )
        fig_scatter.add_trace(
            go.Scatter(
                x=x.sort_index(),
                y=reg_line.sort_index(),
                mode="lines",
                name="Recta CAPM",
                line=dict(color="#ff6b6b", width=2)
            )
        )
        fig_scatter.update_layout(
            title="Exceso de retorno diario: activo vs mercado",
            xaxis_title=f"Exceso mercado ({benchmark})",
            yaxis_title=f"Exceso activo ({ticker})",
            height=420,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        with st.expander("Info del modelo: CAPM y regresión"):
            st.markdown(f"""
**CAPM (Capital Asset Pricing Model)**  
Modelo:  \( R_i - R_f = \\alpha + \\beta (R_m - R_f) + \\varepsilon \)

- \(R_i\): retorno del activo ({ticker}).  
- \(R_m\): retorno del mercado ({benchmark}).  
- \(R_f\): tasa libre de riesgo (aquí: {rf_annual_pct:0.2f} % anual).  

**Interpretación:**  
- **Beta ≈ {beta:0.2f}** → sensibilidad del activo al mercado (riesgo sistemático).  
- **Alpha anual ≈ {alpha_annual*100:0.2f}%** → exceso de retorno promedio vs lo que explicaría el riesgo sistemático.  
- **R² ≈ {r2:0.3f}** → proporción de la variabilidad del exceso de retorno explicada por el mercado.
""")

        # ========= Beta rolling =========
        st.markdown("### Beta rolling")

        returns_for_roll = pd.concat(
            [
                r["asset"].rename("asset"),
                r["bench"].rename("bench")
            ],
            axis=1,
            join="inner"
        ).dropna()

        beta_roll = compute_rolling_beta(returns_for_roll, window=int(beta_window))

        fig_beta = go.Figure()
        fig_beta.add_trace(
            go.Scatter(
                x=beta_roll.index,
                y=beta_roll.values,
                name=f"Beta rolling ({int(beta_window)}d)"
            )
        )
        fig_beta.add_hline(y=1.0, line_dash="dash", annotation_text="Beta = 1")
        fig_beta.update_layout(
            title=f"Beta rolling {ticker} vs {benchmark}",
            height=320,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_beta, use_container_width=True)

        with st.expander("Info del indicador: Beta rolling"):
            st.markdown(f"""
**Beta rolling ({int(beta_window)} días):**  
Calcula la beta en ventanas móviles para ver cómo ha cambiado el riesgo sistemático del activo en el tiempo.

- Beta **> 1** → el activo tiende a amplificar los movimientos del mercado.  
- Beta **< 1** → el activo se mueve menos que el mercado.  
- Cambios bruscos en la beta pueden reflejar eventos de negocio, cambios de régimen o variaciones en la estructura de riesgo.
""")

# =============================
# Frontera Tab
# =============================

with tab_frontier:
    st.subheader("Frontera eficiente y asignación óptima (Markowitz)")
    st.caption(
    "Modelo de optimización de portafolio basado en Markowitz: "
    "generación de miles de combinaciones, estimación de retorno–riesgo y selección del portafolio óptimo "
    "según tu nivel de aversión al riesgo."
    )
    rf_annual = rf_annual_pct / 100.0

    # Definimos el universo de activos para el portafolio
    tickers_port = [benchmark, ticker]
    for e in [extra1, extra2, extra3]:
        if e.strip():
            tickers_port.append(e)

    # Eliminar duplicados manteniendo orden
    seen = set()
    tickers_port = [t for t in tickers_port if not (t in seen or seen.add(t))]

    st.markdown(f"**Activos incluidos en la evaluación:** {', '.join(tickers_port)}")

    # Descarga de precios
    price_dict = {}
    with st.spinner("Descargando precios de todos los activos del portafolio..."):
        for sym in tickers_port:
            df_sym = download_prices(sym, years)
            if not df_sym.empty and "Close" in df_sym.columns:
                price_dict[sym] = df_sym["Close"].dropna()

    if len(price_dict) < 2:
        st.error("Se necesitan al menos 2 activos con datos válidos para construir una frontera.")
        st.stop()

    # Alineamos todas las series de precios por fecha
    prices_df = pd.concat(price_dict.values(), axis=1, join="inner")
    prices_df.columns = list(price_dict.keys())

    # Retornos diarios
    returns = prices_df.pct_change().dropna()

    if returns.empty:
        st.error("No se pudieron obtener retornos suficientes para el portafolio.")
        st.stop()

    # Parámetros básicos: medias y covarianza
    n_assets = returns.shape[1]
    cols = list(returns.columns)

    # Retornos anualizados por activo
    n = len(returns)
    mean_daily = returns.mean()
    mean_annual = (1 + mean_daily) ** 252 - 1

    cov_daily = returns.cov()
    cov_annual = cov_daily * 252

    # =========================
    # Simulación de portafolios aleatorios (curva Markowitz)
    # =========================
    n_portfolios = 8000

    all_weights = np.random.dirichlet(np.ones(n_assets), size=n_portfolios)  # long-only
    port_returns = all_weights @ mean_annual.values
    port_vols = np.sqrt(np.einsum("ij,jk,ik->i", all_weights, cov_annual.values, all_weights))

    # Utilidad de cada portafolio según aversión A
    # U = E[R] - 0.5 * A * σ²
    U = port_returns - 0.5 * risk_aversion * (port_vols ** 2)
    idx_star = np.argmax(U)
    w_star = all_weights[idx_star]
    ret_star = port_returns[idx_star]
    vol_star = port_vols[idx_star]

    # Sharpe de cada portafolio
    sharpe_all = (port_returns - rf_annual) / port_vols

    # =========================
    # Gráfica de frontera simulada
    # =========================

    fig_frontier = go.Figure()

    # Nube de portafolios coloreada por Sharpe
    fig_frontier.add_trace(
        go.Scatter(
            x=port_vols,
            y=port_returns,
            mode="markers",
            name="Portafolios simulados",
            marker=dict(
                size=5,
                color=sharpe_all,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Sharpe")
            ),
            opacity=0.6,
        )
    )

    # Punto óptimo según aversión A
    fig_frontier.add_trace(
        go.Scatter(
            x=[vol_star],
            y=[ret_star],
            mode="markers",
            name=f"Portafolio óptimo (A={risk_aversion:0.1f})",
            marker=dict(size=11, color="#ff6b6b", symbol="diamond")
        )
    )

    # Puntos individuales de cada activo
    asset_vols = np.sqrt(np.diag(cov_annual.values))
    fig_frontier.add_trace(
        go.Scatter(
            x=asset_vols,
            y=mean_annual.values,
            mode="markers+text",
            name="Activos individuales",
            text=[f"{c}" for c in cols],
            textposition="top center",
            marker=dict(size=9, color="#ffaa00"),
        )
    )

    fig_frontier.update_layout(
        title="Retorno vs riesgo del portafolio (frontera simulada)",
        xaxis_title="Riesgo σ (volatilidad anualizada)",
        yaxis_title="Retorno anual esperado",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(fig_frontier, use_container_width=True)

        # =========================
    # Resumen de asignación óptima
    # =========================
    asset_vols = np.sqrt(np.diag(cov_annual.values))  # vol por activo (ya la usas arriba)

    weights_table = pd.DataFrame(
        {
            "Activo": cols,
            "Peso óptimo": w_star,
            "Retorno anual activo": mean_annual.values,
            "Volatilidad anual": asset_vols,
        }
    )

    # Ordenamos por peso descendente para que el “core” quede arriba
    weights_table = weights_table.sort_values("Peso óptimo", ascending=False).reset_index(drop=True)

    # Métricas agregadas del portafolio
    colA, colB, colC = st.columns(3)
    colA.metric("Retorno esperado portafolio óptimo", f"{ret_star*100:0.2f}%")
    colB.metric("Riesgo (volatilidad) portafolio óptimo", f"{vol_star*100:0.2f}%")
    colC.metric("Sharpe portafolio óptimo", f"{((ret_star - rf_annual)/vol_star):0.2f}")

    st.markdown("#### Composición del portafolio óptimo")
    

    # ---------- Tabla estilizada con barras de progreso ----------

    df_display = weights_table.copy()
    df_display["Peso óptimo (%)"] = (df_display["Peso óptimo"] * 100).round(2)
    df_display["Retorno anual activo (%)"] = (df_display["Retorno anual activo"] * 100).round(2)
    df_display["Volatilidad anual (%)"] = (df_display["Volatilidad anual"] * 100).round(2)

    df_display = df_display[
        ["Activo", "Peso óptimo (%)", "Retorno anual activo (%)", "Volatilidad anual (%)"]
    ]

    max_peso = float(df_display["Peso óptimo (%)"].max())

    

    st.dataframe(
        df_display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Activo": st.column_config.TextColumn("Activo"),
            "Peso óptimo (%)": st.column_config.ProgressColumn(
                "Peso óptimo",
                format="%.2f %%",
                min_value=0.0,
                max_value=max_peso,
            ),
            "Retorno anual activo (%)": st.column_config.NumberColumn(
                "Retorno anual activo (%)", format="%.2f %%"
            ),
            "Volatilidad anual (%)": st.column_config.NumberColumn(
                "Volatilidad anual (%)", format="%.2f %%"
            ),
        },
    )



    with st.expander("Info del modelo: Frontera Markowitz y aversión al riesgo"):
        st.markdown(
            f"""
Este bloque construye una **aproximación de la frontera eficiente** para el conjunto de activos:

- {', '.join(cols)}

Utiliza miles de portafolios aleatorios (long-only) para generar la nube de puntos:
cada punto es una combinación lineal de los activos con pesos que suman 1.

**Ejes:**

- Eje X: volatilidad anual del portafolio (riesgo).  
- Eje Y: retorno anual esperado del portafolio.  

La parte **superior izquierda** de la nube se aproxima a la frontera eficiente:
portafolios que maximizan retorno dado un nivel de riesgo.

La **aversión al riesgo A = {risk_aversion:0.1f}** se usa con la función:

> U = E[R_p] − 0.5 · A · σ_p²

y seleccionamos el portafolio con **máxima utilidad U** como portafolio óptimo.

- Si reduces **A**, el modelo preferirá portafolios más agresivos (más riesgo y retorno).  
- Si aumentas **A**, migrará hacia portafolios más defensivos.  
"""
        )

# =============================
# Resto de tabs como placeholders
# =============================

with tab_ai:
    st.subheader("AI Advisor")
    st.caption(
        "Este bloque sintetiza los resultados cuantitativos (CAPM y portafolio Markowitz) "
        "y los envía a un modelo de lenguaje para obtener un análisis narrativo."
    )

    # Badge visual según el modo
    mode_label = "Modo informal" if paps_mode else "Modo formal y profesional"
    mode_color = "#f97316" if paps_mode else "#3b82f6"

    st.markdown(
        f"""
        <div style="
            display:inline-block;
            padding:4px 10px;
            border-radius:999px;
            background-color:{mode_color}20;
            border:1px solid {mode_color};
            color:{mode_color};
            font-size:11px;
            font-weight:600;
            margin-bottom:10px;
        ">
            {mode_label}
        </div>
        """,
        unsafe_allow_html=True,
    )

    rf_annual = rf_annual_pct / 100.0

    # --- Reusar cálculos de CAPM (ticker vs benchmark) ---
    with st.spinner("Preparando datos para el análisis (CAPM + Markowitz)..."):
        data_ticker_ai = download_prices(ticker, years)
        data_bench_ai = download_prices(benchmark, years)

    if data_ticker_ai.empty or data_bench_ai.empty:
        st.error("No se pudieron obtener datos suficientes para el ticker o el benchmark.")
        st.stop()
    else:
        # Cierre alineado
        try:
            close_a_ai = data_ticker_ai["Close"].dropna()
            close_m_ai = data_bench_ai["Close"].dropna()
        except KeyError as e:
            st.error(f"Faltan columnas Close en los datos: {e}")
            st.stop()

        close_a_ai, close_m_ai = close_a_ai.align(close_m_ai, join="inner")

        # Métricas CAPM
        capm_metrics_ai = compute_capm_metrics(close_a_ai, close_m_ai, rf_annual=rf_annual)

        # ---------- Mini-Markowitz para el AI Advisor ----------
        # Reaprovechamos la misma lógica de frontera, pero en pequeño
        tickers_port_ai = [benchmark, ticker]
        for e in [extra1, extra2, extra3]:
            if e.strip():
                tickers_port_ai.append(e.upper())

        seen_ai = set()
        tickers_port_ai = [t for t in tickers_port_ai if not (t in seen_ai or seen_ai.add(t))]

        price_dict_ai = {}
        for sym in tickers_port_ai:
            df_sym = download_prices(sym, years)
            if not df_sym.empty and "Close" in df_sym.columns:
                price_dict_ai[sym] = df_sym["Close"].dropna()

        if len(price_dict_ai) >= 2:
            prices_ai = pd.concat(price_dict_ai.values(), axis=1, join="inner")
            prices_ai.columns = list(price_dict_ai.keys())
            returns_ai = prices_ai.pct_change().dropna()

            n_assets_ai = returns_ai.shape[1]
            mean_daily_ai = returns_ai.mean()
            mean_annual_ai = (1 + mean_daily_ai) ** 252 - 1
            cov_annual_ai = returns_ai.cov() * 252

            n_portfolios_ai = 3000  # un poco más ligero que el bloque de frontera
            all_w = np.random.dirichlet(np.ones(n_assets_ai), size=n_portfolios_ai)
            port_ret = all_w @ mean_annual_ai.values
            port_vol = np.sqrt(np.einsum("ij,jk,ik->i", all_w, cov_annual_ai.values, all_w))
            U_ai = port_ret - 0.5 * risk_aversion * (port_vol ** 2)
            idx_star_ai = np.argmax(U_ai)
            w_star_ai = all_w[idx_star_ai]
            ret_star_ai = port_ret[idx_star_ai]
            vol_star_ai = port_vol[idx_star_ai]
            sharpe_star_ai = (ret_star_ai - rf_annual) / vol_star_ai if vol_star_ai > 0 else np.nan

            markowitz_result_ai = {
                "assets": list(prices_ai.columns),
                "weights": w_star_ai.tolist(),
                "ret_star": float(ret_star_ai),
                "vol_star": float(vol_star_ai),
                "sharpe_star": float(sharpe_star_ai),
                "A": float(risk_aversion),
            }
        else:
            markowitz_result_ai = {
                "assets": [ticker],
                "weights": [1.0],
                "ret_star": capm_metrics_ai["asset_ann"],
                "vol_star": capm_metrics_ai["asset_ann_vol"],
                "sharpe_star": capm_metrics_ai["sharpe"],
                "A": float(risk_aversion),
            }

        # ---------- Resumen visual de números ----------
        st.markdown("### Resumen numérico clave")

        c1, c2, c3 = st.columns(3)
        c1.metric("Beta (ticker)", f"{capm_metrics_ai['beta']:.3f}")
        c2.metric("Retorno anual ticker", f"{capm_metrics_ai['asset_ann']*100:0.2f}%")
        c3.metric("Sharpe realizado", f"{capm_metrics_ai['sharpe'] or 0:0.2f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Retorno anual benchmark", f"{capm_metrics_ai['market_ann']*100:0.2f}%")
        c5.metric("Retorno CAPM esperado", f"{capm_metrics_ai['exp_capm']*100:0.2f}%")
        c6.metric("Alpha anual", f"{capm_metrics_ai['alpha_annual']*100:0.2f}%")

        st.markdown("#### Portafolio óptimo (Markowitz) para el AI Advisor")

        c7, c8, c9 = st.columns(3)
        c7.metric("Retorno portafolio óptimo", f"{markowitz_result_ai['ret_star']*100:0.2f}%")
        c8.metric("Riesgo portafolio óptimo", f"{markowitz_result_ai['vol_star']*100:0.2f}%")
        c9.metric("Sharpe portafolio óptimo", f"{markowitz_result_ai['sharpe_star']:0.2f}")

        # --------- Tabla estilizada de pesos para el AI Advisor ---------
        weights_table_ai = pd.DataFrame(
            {
                "Activo": markowitz_result_ai["assets"],
                "Peso óptimo (%)": [w * 100 for w in markowitz_result_ai["weights"]],
            }
        )

        # Ordenamos por peso descendente
        weights_table_ai = weights_table_ai.sort_values(
            "Peso óptimo (%)", ascending=False
        ).reset_index(drop=True)

        max_peso_ai = float(weights_table_ai["Peso óptimo (%)"].max())

        st.dataframe(
            weights_table_ai,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Activo": st.column_config.TextColumn("Activo"),
                "Peso óptimo (%)": st.column_config.ProgressColumn(
                    "Peso óptimo",
                    format="%.2f %%",
                    min_value=0.0,
                    max_value=max_peso_ai,
                ),
            },
        )



        st.markdown("---")
        st.markdown("### Análisis con IA")

        summary_for_ai = build_ai_summary(
            ticker=ticker,
            benchmark=benchmark,
            rf_annual=rf_annual,
            capm_metrics=capm_metrics_ai,
            markowitz_result=markowitz_result_ai,
        )

        if st.button("Generar análisis con IA", key="ai_generate_button"):
            with st.spinner("Consultando al modelo de IA..."):
                try:
                    analysis_text = call_ai_advisor(summary_for_ai, paps_mode=paps_mode)

                    st.markdown("#### Resultado del AI Advisor")

                    # Card visual para el texto del análisis
                    safe_html = analysis_text.replace("\n", "<br>")

                    st.markdown(
                        f"""
                        <div style="
                            background-color:#020617;
                            border-radius:14px;
                            padding:18px 20px;
                            border:1px solid #1f2937;
                            margin-top:8px;
                            margin-bottom:12px;
                            font-size:14px;
                            line-height:1.55;
                            color:#e5e7eb;
                        ">
                            {safe_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                except Exception as e:
                    st.error(f"Error llamando a la API de OpenAI: {e}")

            st.caption(
                "El análisis se basa únicamente en los datos numéricos de esta app y no constituye "
                "recomendación de inversión personalizada."
            )
