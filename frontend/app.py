import os
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
#page config
st.set_page_config(page_title="NarrativeMap", page_icon="◈", layout="wide")
BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")
CLUSTER_COLORS = {
    0: "#e8c86d",
    1: "#7c6ef5",
    2: "#5ec8a0",
    3: "#e07070",
    4: "#60b4e8",
}
CLUSTER_NAMES = {
    0: "Narrative A",
    1: "Narrative B",
    2: "Narrative C",
    3: "Narrative D",
    4: "Narrative E",
}
SENTIMENT_COLORS = {
    "positive": "#5ec8a0",
    "neutral":  "#888aaa",
    "negative": "#e07070",
}
SENTIMENT_ICONS = {"positive": "▲", "neutral": "●", "negative": "▼"}
#css
def load_css(path: str = "style.css") -> None:
    css_path = os.path.join(os.path.dirname(__file__), path)
    if not os.path.exists(css_path):
        st.warning(f"stylesheet not found: {css_path}")
        return
    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


load_css()
#heler_functions
def fmt_date(raw: str) -> str:
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).strftime("%-d %b %Y")
    except Exception:
        return raw or "—"
def render_nds_card(cid: int, nd: dict) -> str:
    color = CLUSTER_COLORS.get(cid, "#888")
    name  = CLUSTER_NAMES.get(cid, f"Narrative {cid}")
    dom   = nd["dominant"]
    cls   = "nds-card dominant" if dom else "nds-card"
    crown = '<div class="nds-crown">◆ DOMINANT</div>' if dom else ""
    sd    = nd["sentiment_dist"]
    dsent = nd["dominant_sentiment"]
    sc    = SENTIMENT_COLORS.get(dsent, "#888")
    si    = SENTIMENT_ICONS.get(dsent, "●")
    bar_w = min(nd["nds"], 100)
    return (
        f'<div class="{cls}">'
        f'<div class="nds-accent" style="background:{color}"></div>'
        f'{crown}'
        f'<div class="nds-name">{name}</div>'
        f'<div class="nds-score" style="color:{color}">{nd["nds"]:.1f}</div>'
        f'<div class="nds-meta">'
        f'Vol <span>{nd["volume"]}</span> &nbsp;·&nbsp; Growth <span>{nd["growth_rate"]:.2f}/hr</span><br>'
        f'Stability <span>{nd["stability"]:.2f}</span> &nbsp;·&nbsp; Emo.Wt <span>{nd["emotional_weight"]:.2f}</span><br>'
        f'Volatility <span>{nd["volatility"]:.2f}</span><br>'
        f'<span style="color:{sc}">{si} {dsent.title()}</span>'
        f'&nbsp;·&nbsp; pos {sd.get("positive", 0)} neu {sd.get("neutral", 0)} neg {sd.get("negative", 0)}'
        f'</div>'
        f'<div class="nds-bar-track">'
        f'<div class="nds-bar-fill" style="width:{bar_w}%;background:{color}"></div>'
        f'</div>'
        f'</div>'
    )
def render_article_card(art: dict) -> str:
    cid   = art["narrative"]
    color = CLUSTER_COLORS.get(cid, "#888")
    name  = CLUSTER_NAMES.get(cid, f"Cluster {cid}")
    sent  = art.get("sentiment", "neutral")
    sc    = SENTIMENT_COLORS.get(sent, "#888")
    si    = SENTIMENT_ICONS.get(sent, "●")
    conf  = int(art.get("sentiment_conf", 0) * 100)
    url   = art.get("url", "")
    link  = f'<div class="card-link"><a href="{url}" target="_blank">Read →</a></div>' if url else ""
    title = art.get("title") or art["text"][:100]
    return (
        f'<div class="card">'
        f'<div class="card-bar" style="background:{color}"></div>'
        f'<div class="card-meta">'
        f'<span class="card-source">{art["source"]}</span>'
        f'<div class="card-badges">'
        f'<span class="card-badge" style="background:{sc}22;color:{sc}">{si} {sent}</span>'
        f'<span class="card-badge" style="background:{color}22;color:{color}">{name}</span>'
        f'</div></div>'
        f'<div class="card-title">{title}</div>'
        f'<div class="card-text">{art["text"]}</div>'
        f'<div class="conf-track">'
        f'<div class="conf-fill" style="width:{conf}%;background:{sc}"></div>'
        f'</div>'
        f'<div class="card-footer">'
        f'<span class="card-date">{fmt_date(art["publishedAt"])}</span>'
        f'{link}'
        f'</div>'
        f'</div>'
    )


#graph_layout
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", size=11, color="#9090b0"),
    # margin and legend omitted — each chart defines its own to avoid
    # "multiple values for keyword argument" errors when unpacking **PLOTLY_LAYOUT.
)
AXIS_STYLE = dict(gridcolor="#1a1a28", zerolinecolor="#2a2a3a", linecolor="#1e1e2e")

#chat-builder
def build_sentiment_timeline_chart(timeline: dict, nds: dict, topic: str) -> go.Figure:
    fig = go.Figure()

    fig.add_hrect(y0=0.0,  y1=1.1,  fillcolor="rgba(94,200,160,0.07)",  line_width=0)
    fig.add_hrect(y0=-1.1, y1=0.0,  fillcolor="rgba(224,112,112,0.07)", line_width=0)

    for y_val, label, color in [
        (0.85,  "▲ POSITIVE", "#5ec8a0"),
        (-0.85, "▼ NEGATIVE", "#e07070"),
    ]:
        fig.add_annotation(
            x=1.01, y=y_val, xref="paper", yref="y",
            text=label, showarrow=False,
            font=dict(size=9, color=color, family="IBM Plex Mono, monospace"),
            xanchor="left",
        )

    #Neutral zero line
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="#3a3a55",
        line_width=1.5,
        annotation_text="NEUTRAL  ●  0.0",
        annotation_position="bottom left",
        annotation_font=dict(size=9, color="#6a6a8a", family="IBM Plex Mono, monospace"),
    )

    for cid_str, points in timeline.items():
        cid       = int(cid_str)
        color     = CLUSTER_COLORS.get(cid, "#888")
        name      = CLUSTER_NAMES.get(cid, f"Narrative {cid}")
        nds_score = nds.get(cid, {}).get("nds", 0)
        is_dom    = nds.get(cid, {}).get("dominant", False)

        if not points:
            continue

        df = pd.DataFrame(points)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        if df.empty:
            continue

        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

        hover = [
            f"<b>{row['title'][:70]}</b><br>"
            f"──────────────────────<br>"
            f"Sentiment : <b style='color:{color}'>{row['sentiment'].upper()}</b>  {row['sentiment_score']:+.2f}<br>"
            f"Confidence: {row['confidence'] * 100:.0f}%<br>"
            f"Source    : {row['source']}<br>"
            f"NDS       : {nds_score:.1f}{'  ◆ dominant' if is_dom else ''}"
            for _, row in df.iterrows()
        ]

        # Filled area beneath line
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["sentiment_score"],
            fill="tozeroy",
            mode="none",
            fillcolor=f"rgba({r},{g},{b},0.12)",
            showlegend=False,
            hoverinfo="skip",
        ))

        # Main line
        line_width = 3.5 if is_dom else 2.0
        marker_size = 10 if is_dom else 7
        legend_label = f"{'◆ ' if is_dom else ''}{name}  ·  NDS {nds_score:.0f}"

        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["sentiment_score"],
            mode="lines+markers",
            name=legend_label,
            line=dict(color=color, width=line_width, shape="spline", smoothing=1.1),
            marker=dict(
                size=marker_size,
                color=color,
                line=dict(width=2, color=f"rgba({r},{g},{b},0.3)"),
                symbol="circle",
            ),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"Sentiment Shift Over Time — <b>{topic}</b>",
            font=dict(size=15, color="#e8e8f0", family="Playfair Display, serif"),
            x=0, y=0.97,
        ),
        yaxis=dict(
            **AXIS_STYLE,
            title=dict(text="Sentiment Score", font=dict(size=11, color="#6a6a8a")),
            range=[-1.15, 1.15],
            tickvals=[-1.0, -0.5, 0.0, 0.5, 1.0],
            ticktext=["−1.0  Negative", "−0.5", "0.0  Neutral", "+0.5", "+1.0  Positive"],
            tickfont=dict(size=10, color="#7a7a9a"),
            showgrid=True,
            gridwidth=1,
        ),
        xaxis=dict(
            **AXIS_STYLE,
            title=dict(text="Publication Date", font=dict(size=11, color="#6a6a8a")),
            tickformat="%b %d",
            tickangle=-35,
            tickfont=dict(size=10, color="#7a7a9a"),
            showgrid=True,
            gridwidth=1,
            nticks=12,
        ),
        height=500,
        hovermode="closest",
        legend=dict(
            bgcolor="rgba(10,10,20,0.85)",
            bordercolor="#2e2e45",
            borderwidth=1,
            font=dict(size=11, color="#c0c0d8", family="IBM Plex Mono, monospace"),
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
            tracegroupgap=4,
        ),
        margin=dict(l=20, r=80, t=50, b=60),
    )
    return fig


def build_nds_bar_chart(nds: dict, topic: str) -> go.Figure:
    sorted_ids = sorted(nds.keys(), key=lambda c: nds[c]["nds"], reverse=True)
    names   = [CLUSTER_NAMES.get(c, f"Narrative {c}") for c in sorted_ids]
    scores  = [nds[c]["nds"]        for c in sorted_ids]
    colors  = [CLUSTER_COLORS.get(c, "#888") for c in sorted_ids]
    vols    = [nds[c]["volume"]      for c in sorted_ids]
    growths = [nds[c]["growth_rate"] for c in sorted_ids]
    stabs   = [nds[c]["stability"]   for c in sorted_ids]
    emos    = [nds[c]["emotional_weight"] for c in sorted_ids]
    dom     = [nds[c]["dominant"]    for c in sorted_ids]
    dsents  = [nds[c]["dominant_sentiment"] for c in sorted_ids]

    sent_icon = {"positive": "▲", "neutral": "●", "negative": "▼"}

    hover = [
        f"<b>{'◆ ' if dom[i] else ''}{names[i]}</b><br>"
        f"──────────────────<br>"
        f"NDS Score  : <b>{scores[i]:.1f} / 100</b><br>"
        f"Volume     : {vols[i]} articles<br>"
        f"Growth     : {growths[i]:.2f} art/hr<br>"
        f"Stability  : {stabs[i]:.2f}<br>"
        f"Emo. Weight: {emos[i]:.2f}<br>"
        f"Sentiment  : {sent_icon.get(dsents[i], '●')} {dsents[i].title()}"
        for i in range(len(sorted_ids))
    ]

    fig = go.Figure()

    # Ghost track (full 100-wide background bar) so scale is always visible
    fig.add_trace(go.Bar(
        y=names, x=[100] * len(names), orientation="h",
        marker=dict(color="rgba(255,255,255,0.03)", line=dict(width=0)),
        showlegend=False, hoverinfo="skip",
    ))

    # Main bars
    bar_colors = []
    for i, c in enumerate(colors):
        r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
        alpha = 0.95 if dom[i] else 0.70
        bar_colors.append(f"rgba({r},{g},{b},{alpha})")

    fig.add_trace(go.Bar(
        y=names, x=scores, orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover,
        text=[f"  {s:.1f}" for s in scores],
        textposition="inside",
        textfont=dict(
            color=["#0a0a0f" if dom[i] else "#e8e8f0" for i in range(len(sorted_ids))],
            size=12,
            family="IBM Plex Mono, monospace",
        ),
        showlegend=False,
    ))

    # Dominant marker — gold left pip
    for i, (is_dom, nm) in enumerate(zip(dom, names)):
        if is_dom:
            fig.add_annotation(
                x=scores[i] + 3, y=nm,
                text="◆ DOMINANT",
                showarrow=False,
                font=dict(size=10, color="#e8c86d", family="IBM Plex Mono, monospace"),
                xanchor="left",
            )

    row_h = 72
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="overlay",
        title=dict(
            text=f"Narrative Dominance Score — <b>{topic}</b>",
            font=dict(size=15, color="#e8e8f0", family="Playfair Display, serif"),
            x=0,
        ),
        xaxis=dict(
            **AXIS_STYLE,
            title=dict(text="NDS  (0 – 100)", font=dict(size=11, color="#6a6a8a")),
            range=[0, 130],
            tickvals=[0, 25, 50, 75, 100],
            tickfont=dict(size=10, color="#7a7a9a"),
        ),
        yaxis=dict(
            **AXIS_STYLE,
            title="",
            autorange="reversed",
            tickfont=dict(size=12, color="#c0c0d8", family="IBM Plex Mono, monospace"),
        ),
        height=80 + len(sorted_ids) * row_h,
        showlegend=False,
        bargap=0.30,
        margin=dict(l=20, r=20, t=50, b=50),
    )
    return fig


def build_sentiment_distribution_chart(nds: dict, topic: str) -> go.Figure:

    sorted_ids = sorted(nds.keys(), key=lambda c: nds[c]["nds"], reverse=True)
    names = [CLUSTER_NAMES.get(c, f"Narrative {c}") for c in sorted_ids]

    def pct(cid, label):
        sd  = nds[cid]["sentiment_dist"]
        tot = sum(sd.values()) or 1
        return round(sd.get(label, 0) / tot * 100, 1)

    def cnt(cid, label):
        return nds[cid]["sentiment_dist"].get(label, 0)

    seg_cfg = [
        ("positive", "#5ec8a0", "▲ Positive"),
        ("neutral",  "#6a6a9a", "● Neutral"),
        ("negative", "#e07070", "▼ Negative"),
    ]

    fig = go.Figure()
    for key, color, label in seg_cfg:
        pcts   = [pct(c, key) for c in sorted_ids]
        counts = [cnt(c, key) for c in sorted_ids]
        texts  = [f"{p:.0f}%" if p >= 8 else "" for p in pcts]
        hover  = [
            f"<b>{label}</b><br>{counts[i]} articles  ({pcts[i]:.1f}%)"
            for i in range(len(sorted_ids))
        ]
        fig.add_trace(go.Bar(
            name=label,
            y=names,
            x=pcts,
            orientation="h",
            marker=dict(color=color, line=dict(width=0)),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
            text=texts,
            textposition="inside",
            textfont=dict(size=10, color="#0a0a0f", family="IBM Plex Mono, monospace"),
        ))

    row_h = 72
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="stack",
        title=dict(
            text=f"Sentiment Composition — <b>{topic}</b>",
            font=dict(size=15, color="#e8e8f0", family="Playfair Display, serif"),
            x=0,
        ),
        xaxis=dict(
            **AXIS_STYLE,
            title=dict(text="Share (%)", font=dict(size=11, color="#6a6a8a")),
            range=[0, 100],
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0%", "25%", "50%", "75%", "100%"],
            tickfont=dict(size=10, color="#7a7a9a"),
            showgrid=True,
            gridwidth=1,
        ),
        yaxis=dict(
            **AXIS_STYLE,
            title="",
            autorange="reversed",
            tickfont=dict(size=12, color="#c0c0d8", family="IBM Plex Mono, monospace"),
        ),
        height=80 + len(sorted_ids) * row_h,
        bargap=0.30,
        legend=dict(
            bgcolor="rgba(10,10,20,0.85)",
            bordercolor="#2e2e45",
            borderwidth=1,
            font=dict(size=11, color="#c0c0d8", family="IBM Plex Mono, monospace"),
            orientation="h",
            x=0, y=1.08,
        ),
        margin=dict(l=20, r=20, t=60, b=50),
    )
    return fig


#header
st.markdown("""
<div class="nm-header">
  <div class="nm-title">Narrative<span>Map</span></div>
  <div class="nm-sub">News Intelligence · GDELT Sentiment Model · NDS Engine · Google RSS</div>
</div>
""", unsafe_allow_html=True)

#search bar
col_q, col_k, col_btn = st.columns([4, 1, 1])
with col_q:
    topic = st.text_input(
        label="topic", label_visibility="collapsed",
        placeholder="Search topic — AI, climate, markets, war…",
        key="topic_input",
    )
with col_k:
    n_clusters = st.selectbox("Narratives", options=[2, 3, 4, 5], index=1)
with col_btn:
    search = st.button("Analyse →")


# main
if search and topic.strip():
    with st.spinner("Fetching articles via Google News RSS · running GDELT sentiment model · computing NDS & timeline…"):
        try:
            resp = requests.get(
                f"{BACKEND}/news",
                params={"topic": topic.strip(), "n_clusters": n_clusters},
                timeout=90,
            )
            data = resp.json()
        except requests.exceptions.ConnectionError:
            data = {"error": (
                "Cannot connect to FastAPI backend.\n\n"
                "Start the server:\n"
                "  cd backend\n"
                "  uvicorn main:app --reload\n\n"
                "First-time setup:\n"
                "  python save_model.py\n"
                "  python train_gdelt.py\n\n"
                "No API key required — articles fetched from Google News RSS."
            )}
        except requests.exceptions.Timeout:
            data = {"error": "Request timed out (90 s). Backend may still be loading the model on first run."}
        except Exception as e:
            data = {"error": str(e)}

    if "error" in data:
        st.markdown(f'<div class="err-box">⚠ {data["error"]}</div>', unsafe_allow_html=True)

    else:
        articles    = data["articles"]
        nds         = {int(k): v for k, v in data["nds"].items()}
        timeline    = data.get("sentiment_timeline", {})
        dominant_id = int(data["dominant_narrative"])
        n_actual    = len(nds)

        pos_count = sum(1 for a in articles if a.get("sentiment") == "positive")
        neg_count = sum(1 for a in articles if a.get("sentiment") == "negative")
        neu_count = sum(1 for a in articles if a.get("sentiment") == "neutral")
        dom_name  = CLUSTER_NAMES.get(dominant_id, f"Narrative {dominant_id}")
        dom_score = nds[dominant_id]["nds"]

        # stats
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-box">
            <div class="stat-num">{data['total_articles']}</div>
            <div class="stat-lbl">Articles</div>
          </div>
          <div class="stat-box">
            <div class="stat-num">{n_actual}</div>
            <div class="stat-lbl">Narratives</div>
          </div>
          <div class="stat-box">
            <div class="stat-num" style="color:#5ec8a0">{pos_count}</div>
            <div class="stat-lbl">Positive</div>
          </div>
          <div class="stat-box">
            <div class="stat-num" style="color:#888aaa">{neu_count}</div>
            <div class="stat-lbl">Neutral</div>
          </div>
          <div class="stat-box">
            <div class="stat-num" style="color:#e07070">{neg_count}</div>
            <div class="stat-lbl">Negative</div>
          </div>
          <div class="stat-box">
            <div class="stat-num" style="font-size:1.05rem;padding-top:0.55rem">{dom_name}</div>
            <div class="stat-lbl">Dominant · NDS {dom_score:.1f}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        #nds card
        st.markdown('<div class="section-title">Narrative Dominance Score (NDS)</div>',
                    unsafe_allow_html=True)
        ranked_ids = sorted(nds.keys(), key=lambda c: nds[c]["rank"])
        cards_html = "".join(render_nds_card(cid, nds[cid]) for cid in ranked_ids)
        st.markdown(f'<div class="nds-row">{cards_html}</div>', unsafe_allow_html=True)

        # sentiment shift timeline
        st.markdown('<div class="section-title">Narrative Sentiment Shift Over Time</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.78rem;color:#5a5a7a;margin-bottom:0.8rem'>"
            "Each line shows how a narrative's sentiment score (+1 = positive, −1 = negative) "
            "evolves across publication time. Scored by the GDELT-trained sentiment model.</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_sentiment_timeline_chart(timeline, nds, topic.strip()),
            use_container_width=True, config={"displayModeBar": False},
        )

        # nds+sentiment distribution
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown('<div class="section-title">NDS Score per Narrative</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(
                build_nds_bar_chart(nds, topic.strip()),
                use_container_width=True, config={"displayModeBar": False},
            )
        with col_right:
            st.markdown('<div class="section-title">Sentiment Composition</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(
                build_sentiment_distribution_chart(nds, topic.strip()),
                use_container_width=True, config={"displayModeBar": False},
            )

        #article feed
        st.markdown('<div class="section-title">Article Feed</div>', unsafe_allow_html=True)
        dots = "".join(
            f'<div class="legend-item">'
            f'<span class="legend-dot" style="background:{CLUSTER_COLORS[i]}"></span>'
            f'{CLUSTER_NAMES[i]}</div>'
            for i in range(n_actual)
        )
        st.markdown(
            f'<div class="legend"><span class="legend-lbl">Clusters →</span>{dots}</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        for i, article in enumerate(articles):
            with cols[i % 3]:
                st.markdown(render_article_card(article), unsafe_allow_html=True)

elif search and not topic.strip():
    st.warning("Please enter a topic to search.")

else:
    st.markdown("""
    <div class="empty-state">
      <div class="icon">◈</div>
      <p>Enter a topic to map the narrative shift</p>
    </div>
    """, unsafe_allow_html=True)
