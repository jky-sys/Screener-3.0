import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
import plotly.graph_objects as go

# ==============================================================================
# 配置与页面设置
# ==============================================================================
st.set_page_config(page_title="Trinity Pro V3.0", page_icon="🇨🇳", layout="wide")

# [美股定制列表]
CUSTOM_TICKERS = [
    # === 半导体/芯片 ===
    "NVDA", "AMD", "TSM", "AVGO", "INTC", "QCOM", "MU", "TXN", "AMAT", "ASML", "ARM", "SMCI",
    # === 航天/军工 ===
    "RKLB", "SPCE", "LUNR", "BA", "LMT", "RTX", "PLTR",
    # === 加密/科技 ===
    "MSTR", "COIN", "MARA", "TSLA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", 
    "NET", "SNOW", "PLTR", "AI",
    # === 核能 ===
    "SMR", "OKLO", "CCJ", "CEG", "VST"
]

# [A股热门精选] (注意后缀: .SS=上海, .SZ=深圳)
ASHARES_TICKERS = [
    "600519.SS", # 贵州茅台
    "300750.SZ", # 宁德时代
    "002594.SZ", # 比亚迪
    "601318.SS", # 中国平安
    "600036.SS", # 招商银行
    "601888.SS", # 中国中免
    "000858.SZ", # 五粮液
    "000568.SZ", # 泸州老窖
    "300059.SZ", # 东方财富
    "600276.SS", # 恒瑞医药
    "603288.SS", # 海天味业
    "002475.SZ", # 立讯精密
    "601012.SS", # 隆基绿能
    "002371.SZ", # 北方华创
    "600900.SS", # 长江电力
    "601899.SS", # 紫金矿业
    "000333.SZ", # 美的集团
    "601988.SS", # 中国银行
    "600028.SS", # 中国石化
    "002230.SZ", # 科大讯飞
    "603986.SS", # 兆易创新
    "600522.SS", # 中天科技
    "600150.SS"  # 中国船舶
]

# 纳指/标普备份列表 (简化版以节省空间，逻辑不变)
NAS100_FALLBACK_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "GOOGL", "AMD", "QCOM", "INTC", "CSCO", "PEP", "AVGO", "COST", "TMUS"]
SP500_FALLBACK_TICKERS = ["MSFT", "AAPL", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "LLY", "JPM", "TSLA", "XOM", "UNH", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "KO"]

# ==============================================================================
# 核心逻辑函数
# ==============================================================================
@st.cache_data(ttl=3600)
def get_stock_list(mode):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        if mode == "A_SHARES": return ASHARES_TICKERS # 直接返回A股列表
        if mode == "NAS100":
            url = "https://en.wikipedia.org/wiki/Nasdaq-100"
            df = pd.read_html(io.StringIO(requests.get(url, headers=headers).text))[0]
            col = 'Symbol' if 'Symbol' in df.columns else 'Ticker'
            return list(set([t.replace('.', '-') for t in df[col].tolist()]))
        elif mode == "SP500":
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            df = pd.read_html(io.StringIO(requests.get(url, headers=headers).text))[0]
            return list(set([t.replace('.', '-') for t in df['Symbol'].tolist()]))
        else:
            return CUSTOM_TICKERS
    except:
        return CUSTOM_TICKERS # 兜底

def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_trinity_indicators(df):
    # NX Channels
    df['nx_up1'] = calculate_ema(df['High'], 26)
    df['nx_dw1'] = calculate_ema(df['Low'], 26)
    df['nx_rising'] = (df['nx_up1'] > df['nx_up1'].shift(1)) & (df['nx_dw1'] > df['nx_dw1'].shift(1))
    
    # MACD
    fast_ema = calculate_ema(df['Close'], 12)
    slow_ema = calculate_ema(df['Close'], 26)
    df['dif'] = fast_ema - slow_ema
    df['dea'] = calculate_ema(df['dif'], 9)
    df['macd_gold_cross'] = (df['dif'] > df['dea']) & (df['dif'].shift(1) < df['dea'].shift(1))

    # CD Divergence
    min_price_60 = df['Low'].rolling(60).min()
    min_dif_60 = df['dif'].rolling(60).min()
    price_is_low = df['Low'] <= min_price_60 * 1.05
    dif_is_stronger = df['dif'] > min_dif_60 + 0.1
    df['cd_potential'] = price_is_low & dif_is_stronger & df['macd_gold_cross']

    # INST
    if len(df) < 250:
        df['inst_buy'] = 0
        return df
    
    def rma(series, length): return series.ewm(alpha=1/length, adjust=False).mean()
    
    # 简化版 INST 逻辑
    high_long = df['High'].rolling(250).max()
    low_long  = df['Low'].rolling(250).min()
    low_diff = df['Low'] - df['Low'].shift(1)
    instc = rma(low_diff.abs(), 3) / rma(low_diff.clip(lower=0), 3).replace(0, np.nan) * 100
    instc = instc.fillna(0)
    is_oversold = df['Low'] <= df['Low'].rolling(30).min()
    inst_signal = np.where(is_oversold, instc, 0)
    df['inst_buy'] = calculate_ema(pd.Series(inst_signal, index=df.index), 3)
    
    return df

# ==============================================================================
# 绘图与信息获取
# ==============================================================================
def create_chart(df, ticker):
    plot_df = df.iloc[-150:]
    fig = go.Figure()

    # K线
    fig.add_trace(go.Candlestick(
        x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
        low=plot_df['Low'], close=plot_df['Close'], name='K线'
    ))

    # NX通道
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['nx_up1'], mode='lines', line=dict(color='rgba(41, 98, 255, 0.5)', width=1), name='NX上沿'))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['nx_dw1'], mode='lines', line=dict(color='rgba(41, 98, 255, 0.5)', width=1), name='NX下沿'))

    # 信号标记
    inst_signals = plot_df[plot_df['inst_buy'] > 0.5]
    if not inst_signals.empty:
        fig.add_trace(go.Scatter(x=inst_signals.index, y=inst_signals['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=10, color='#00e676'), name='INST吸筹'))
    
    cd_signals = plot_df[plot_df['cd_potential']]
    if not cd_signals.empty:
        fig.add_trace(go.Scatter(x=cd_signals.index, y=cd_signals['Low']*0.96, mode='markers', marker=dict(symbol='circle', size=8, color='red'), name='CD背离'))

    fig.update_layout(title=f"{ticker} 技术图表", xaxis_rangeslider_visible=False, height=450, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
    return fig

# ==============================================================================
# 主界面逻辑
# ==============================================================================
st.title("🛰️ Trinity Pro: 全球市场雷达 V3.0")
st.markdown("---")

# 侧边栏
st.sidebar.header("📡 扫描配置")
scan_mode = st.sidebar.selectbox("选择市场板块", ["A_SHARES (热门A股)", "CUSTOM (美股科技/核能)", "NAS100 (纳指100)", "SP500 (标普500)"])
period = st.sidebar.selectbox("数据回溯", ["2y", "5y"], index=0)

mode_map = {"A_SHARES (热门A股)": "A_SHARES", "CUSTOM (美股科技/核能)": "CUSTOM", "NAS100 (纳指100)": "NAS100", "SP500 (标普500)": "SP500"}
current_mode = mode_map[scan_mode]

if st.button("🚀 启动扫描", type="primary"):
    tickers = get_stock_list(current_mode)
    st.info(f"正在扫描 {len(tickers)} 只标的，A股数据可能稍慢，请耐心等待...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []
    
    for i, ticker in enumerate(tickers):
        progress_bar.progress((i + 1) / len(tickers))
        status_text.text(f"分析中: {ticker} ...")
        
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval="1d", auto_adjust=True)
            if df.empty or len(df) < 200: continue
            
            df = calculate_trinity_indicators(df)
            curr = df.iloc[-1]
            
            # 筛选逻辑
            recent_accumulation = df['inst_buy'].iloc[-90:].max() > 0.5
            recent_trend_days = df['nx_rising'].iloc[-12:]
            trend_just_started = curr['nx_rising'] and (not recent_trend_days.all())
            has_momentum = df['cd_potential'].iloc[-10:].any() or df['macd_gold_cross'].iloc[-5:].any()
            
            if recent_accumulation and trend_just_started and has_momentum:
                score = 0
                if df['cd_potential'].iloc[-5:].any(): score += 2
                if curr['inst_buy'] > 0.5: score += 1
                
                # 只有当选中时，才获取基本面信息 (节省流量)
                info = {}
                try:
                    # 只有美股和A股支持 info 比较好
                    info = stock.info
                except:
                    info = {}
                
                results.append({
                    "Ticker": ticker,
                    "Price": curr['Close'],
                    "Score": score,
                    "Msg": "双底雏形" + (" + CD背离" if score >=2 else ""),
                    "Data": df,
                    "Info": info, # 存储基本面
                    "StockObj": stock # 存储对象以便获取新闻
                })
                
        except Exception:
            continue

    progress_bar.empty()
    status_text.empty()
    
    if results:
        st.success(f"扫描完成！共发现 {len(results)} 个机会")
        
        for res in results:
            ticker_display = res['Ticker'].replace('.SS', ' (沪)').replace('.SZ', ' (深)')
            
            with st.expander(f"📊 {ticker_display} - ¥/${res['Price']:.2f} | {res['Msg']}"):
                # 使用标签页分隔功能
                tab1, tab2, tab3 = st.tabs(["📈 技术图表", "🏢 基本面概况", "📰 最新新闻"])
                
                with tab1:
                    st.plotly_chart(create_chart(res['Data'], res['Ticker']), use_container_width=True)
                
                with tab2:
                    info = res['Info']
                    if info:
                        col1, col2, col3 = st.columns(3)
                        # 市值 (自动处理单位)
                        mkt_cap = info.get('marketCap', 0)
                        pe_ratio = info.get('trailingPE', 'N/A')
                        
                        col1.metric("市值", f"{mkt_cap/100000000:.2f}亿")
                        col2.metric("市盈率 (PE)", pe_ratio)
                        col3.metric("52周最高", info.get('fiftyTwoWeekHigh', 'N/A'))
                        
                        st.markdown("**公司简介:**")
                        # 简介如果是英文，A股通常没有或者也是英文，这里直接展示
                        st.write(info.get('longBusinessSummary', info.get('longName', '暂无简介')))
                        
                        st.markdown(f"**行业:** {info.get('industry', 'N/A')} | **板块:** {info.get('sector', 'N/A')}")
                    else:
                        st.warning("暂无基本面数据")

                with tab3:
                    st.markdown("##### 最新相关新闻")
                    try:
                        news_list = res['StockObj'].news
                        if news_list:
                            for n in news_list[:5]: # 只显示前5条
                                # 转换时间
                                pub_time = datetime.fromtimestamp(n.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M') if 'providerPublishTime' in n else ""
                                st.markdown(f"**[{n['title']}]({n['link']})**")
                                st.caption(f"发布时间: {pub_time} | 来源: {n.get('publisher', 'Unknown')}")
                                st.markdown("---")
                        else:
                            st.info("暂无最新新闻")
                    except:
                        st.info("获取新闻失败")

    else:
        st.warning("本次扫描未发现符合条件的标的。")