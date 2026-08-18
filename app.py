"""
app.py

StudyMate AI -- Streamlit front end (UI/UX-refreshed).

Landing page (marketing) -> Dashboard (sidebar-nav SaaS shell) on top of the
existing RAG pipeline. See the section headers below; this file is organized
so it can be split into pages/components later if the project grows past a
single file.

Functional core (unchanged)
---------------------------
Sidebar upload -> process_documents -> VectorStoreService -> RetrieverService
-> build_rag_chain / build_notes_chain / build_flashcard_chain / build_quiz_chain.
Exam Mode still reuses the Quiz Generator's schema/chain; it only adds
timing + a locked question set + a scored review.

Scope notes (Streamlit constraints, read before assuming something's a bug)
----------------------------------------------------------------------------
- No true touch/swipe gestures or global Ctrl+K key capture -- those need a
  custom JS component. Flashcards flip via click; sidebar nav has a
  "quick jump" search select as the closest equivalent to a command palette.
- Everything lives in this one app.py to match the project's real structure.
"""

import datetime as _dt
import html
import random
import time
from pathlib import Path

import streamlit as st

from chains.rag_chain import build_rag_chain
from chains.study_chains import build_flashcard_chain, build_notes_chain, build_quiz_chain
from config import (
    DEFAULT_FLASHCARD_COUNT,
    DEFAULT_NOTES_STYLE,
    DEFAULT_QUIZ_DIFFICULTY,
    DEFAULT_QUIZ_QUESTION_COUNT,
    NOTES_STYLES,
    QUIZ_DIFFICULTIES,
    UPLOAD_FOLDER,
)
from llm.huggingface_llm import generative_llm, llm
from pipeline import process_documents
from retriever.retriever import RetrieverService
from vectorstore.faiss_store import VectorStoreService

st.set_page_config(page_title="StudyMate AI", page_icon="📚", layout="wide")


# ============================================================
# Design tokens
#   #F7F7F7 light surface / #C8CED6 mist / #D4CAC5 warm sand
#   #736A86 accent (muted plum) / #272A3B dark surface (ink navy)
# ============================================================

def inject_design_system():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --sm-light:   #ACACB6;
            --sm-mist:    #C8CED6;
            --sm-sand:    #D4CAC5;
            --sm-accent:  #736A86;
            --sm-accent-soft: #8B7FA0;
            --sm-ink:     #272A3B;
            --sm-ink-soft: #3A3752;
            --sm-text:    #2B2B3A;
            --sm-text-muted: #5C5870;
            --sm-glass: rgba(255, 255, 255, 0.66);
            --sm-glass-border: rgba(255, 255, 255, 0.55);
            --sm-radius: 20px;
            --sm-success: #2E8557;
            --sm-success-soft: rgba(58, 168, 107, 0.16);
            --sm-error: #C14545;
            --sm-error-soft: rgba(214, 79, 79, 0.14);
            --sm-warning: #C99042;
            --sm-select: #4D4D59;
            --sm-select-soft: rgba(77, 77, 89, 0.28);
        }

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--sm-text); }
        h1, h2, h3, h4, .sm-display { font-family: 'Plus Jakarta Sans', sans-serif; }

        #MainMenu, header[data-testid="stHeader"] { background: transparent; }
        footer { visibility: hidden; }

        .stApp {
            background:
              radial-gradient(1200px 600px at 85% -10%, rgba(115,106,134,0.18), transparent 60%),
              radial-gradient(900px 500px at 0% 100%, rgba(212,202,197,0.30), transparent 55%),
              linear-gradient(180deg, var(--sm-light) 0%, var(--sm-mist) 48%, var(--sm-accent) 100%);
            background-attachment: fixed;
        }

        @keyframes smFadeInUp { from { opacity:0; transform: translateY(16px);} to {opacity:1; transform: translateY(0);} }
        @keyframes smFadeIn   { from { opacity:0; } to { opacity:1; } }
        @keyframes smFloat    { 0%,100% { transform: translateY(0px);} 50% { transform: translateY(-8px);} }
        @keyframes smDrift    { 0% {transform: translate(0,0);} 50% {transform: translate(18px,-14px);} 100% {transform: translate(0,0);} }
        @keyframes smPulseRing{ 0%,100% {box-shadow:0 0 0 0 rgba(115,106,134,0.35);} 50% {box-shadow:0 0 0 10px rgba(115,106,134,0);} }
        @keyframes smDash     { to { stroke-dashoffset: -24; } }
        @keyframes smBounce   { 0%,80%,100% { transform: translateY(0); opacity:.5;} 40% { transform: translateY(-6px); opacity:1;} }
        @keyframes smPop      { from { opacity:0; transform: scale(.6);} to { opacity:1; transform: scale(1);} }
        @keyframes smShimmer  { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
        @keyframes smGrow     { from { width: 0%; } }

        /* ---------------- Centered page heading ---------------- */
        .sm-section-title {
            font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.75rem; color: var(--sm-ink);
            text-align:center; width:100%; margin: 0.2rem auto 0.2rem auto; letter-spacing:-0.01em;
            animation: smFadeInUp 0.4s ease both;
        }
        .sm-section-title .accent {
            background: linear-gradient(120deg, var(--sm-accent), var(--sm-accent-soft));
            -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
        }
        .sm-section-sub {
            color: var(--sm-text-muted); font-size:0.98rem; text-align:center;
            width:100%; margin: 0 auto 1.4rem auto; max-width: 640px; line-height:1.55;
            animation: smFadeInUp 0.5s ease both;
        }
        .sm-title-rule {
            width: 54px; height: 4px; border-radius: 999px; margin: 0.5rem auto 1.3rem auto;
            background: linear-gradient(90deg, var(--sm-accent), var(--sm-ink));
            animation: smGrow 0.6s ease both;
        }

        /* ---------------- Landing page ---------------- */
        .sm-hero-land {
            position: relative;
            background: linear-gradient(135deg, var(--sm-ink) 0%, var(--sm-accent) 58%, var(--sm-ink) 100%);
            border-radius: 30px; padding:1rem 1rem; text-align: center; overflow: hidden;
            margin: 0rem 0 2rem 0; animation: smFadeInUp 0.9s ease both;
            box-shadow: 0 30px 60px -28px rgba(39,42,59,0.55);
        }
        .sm-blob {
            position: absolute; border-radius: 50%; opacity: 0.30; filter: blur(8px);
            background: radial-gradient(circle, #ffffff, transparent 70%);
            animation: smDrift 9s ease-in-out infinite;
        }
        .sm-eyebrow {
            position: relative; z-index:2; display:inline-block; color:#fff; font-family:'Inter',sans-serif;
            font-weight:600; font-style:normal; font-size:3.00rem;
            padding: 0.4rem 1.2rem; border-radius:999px;
            margin-bottom: 1.4rem; backdrop-filter: blur(6px);
        }
        .sm-hero-land h1 {
            color: #fff; font-weight: 800; font-size: 3.1rem; line-height: 1.15;
            margin: 0 auto 1rem auto; max-width: 760px; position: relative; z-index: 2; letter-spacing: -0.025em;
        }
        .sm-hero-land h1 .grad {
            background: linear-gradient(120deg, #F4EFE9, #D4CAC5);
            -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
        }
        .sm-hero-land h3 .grad {
            background: linear-gradient(120deg, #F4EFE9, #D4CAC5);
            -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
        }
        .sm-hero-land p {
            color: rgba(255,255,255,0.9); font-size: 1.12rem; max-width: 580px; line-height:1.6;
            margin: 0 auto 1.9rem auto; position: relative; z-index: 2;
        }
        .sm-hero-stats {
            position: relative; z-index:2; display:flex; gap:2.4rem; justify-content:center; flex-wrap:wrap;
            margin-top: 2.2rem; padding-top: 1.6rem; border-top: 1px solid rgba(255,255,255,0.18);
        }
        .sm-hero-stats .hs-num { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.5rem; color:#fff; }
        .sm-hero-stats .hs-lbl { color: rgba(255,255,255,0.75); font-size:0.78rem; letter-spacing:0.06em; text-transform:uppercase; }

        .sm-feat {
            text-align:left; padding: 1.5rem 1.4rem; border-radius: 18px; height:100%;
            background: var(--sm-glass); border: 1px solid var(--sm-glass-border); backdrop-filter: blur(14px);
            box-shadow: 0 12px 28px -18px rgba(39,42,59,0.3);
            transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
            animation: smFadeInUp 0.5s ease both;
        }
        .sm-feat:hover { transform: translateY(-5px); box-shadow: 0 18px 34px -16px rgba(115,106,134,0.42); border-color: var(--sm-accent); }
        .sm-feat .ico { font-size:1.7rem; margin-bottom:0.5rem; display:inline-block; animation: smFloat 4s ease-in-out infinite; }
        .sm-feat h3 { margin:0 0 0.3rem 0; color: var(--sm-ink); font-size:1.05rem; font-weight:700; }
        .sm-feat p { margin:0; color: var(--sm-text-muted); font-size:0.88rem; line-height:1.55; }

        .sm-how-wrap { margin: 1rem 0 2.4rem 0; position: relative; }
        .sm-how-line {
            position:absolute; top:24px; left:12%; right:12%; height:2px;
            background: repeating-linear-gradient(90deg, var(--sm-accent) 0 10px, transparent 10px 18px);
            z-index:0;
        }
        .sm-step { text-align:center; padding: 0 0.5rem; position:relative; z-index:1; animation: smFadeInUp 0.5s ease both;}
        .sm-step-num {
            width:48px; height:48px; border-radius:50%; margin:0 auto 0.8rem auto;
            background: linear-gradient(135deg, var(--sm-accent), var(--sm-ink));
            color:#fff; display:flex; align-items:center; justify-content:center;
            font-weight:700; font-family:'Plus Jakarta Sans',sans-serif; font-size:1.05rem;
            animation: smPulseRing 2.6s ease-in-out infinite; border: 3px solid var(--sm-light);
        }
        .sm-step p { color:"#C5C5CC"; font-size: 0.88rem; margin:0.3rem 0 0 0; line-height:1.5; }
        .sm-step strong { color: var(--sm-ink); font-size: 1rem; display:block; margin-bottom:0.15rem; }

        .sm-cta-band {
            background: linear-gradient(120deg, var(--sm-ink), var(--sm-ink-soft));
            border-radius: 26px; text-align:center; padding: 2.8rem 2rem; margin: 1rem 0 2rem 0;
            animation: smFadeInUp 0.6s ease both; box-shadow: 0 24px 50px -26px rgba(39,42,59,0.5);
        }
        .sm-cta-band h2 { color:#fff; font-size:1.7rem; max-width:580px; margin:0 auto 0.4rem auto; letter-spacing:-0.01em; }
        .sm-cta-band p { color: rgba(255,255,255,0.7); font-size:0.95rem; margin:0; }

        /* ---------------- Sidebar ---------------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--sm-ink) 0%, #201E2E 100%);
        }
        section[data-testid="stSidebar"] .sm-side-logo,
        section[data-testid="stSidebar"] .sm-side-caption,
        section[data-testid="stSidebar"] hr { color: #EDEBF7; border-color: rgba(255,255,255,0.15); }
        section[data-testid="stSidebar"] .sm-side-logo {
            font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.3rem; color:#fff;
            margin-bottom: 0.2rem; animation: smFadeIn 0.5s ease both; text-align:center; letter-spacing:0.02em;
        }
        section[data-testid="stSidebar"] .sm-side-tag {
            font-size:0.7rem; color:#B7B3CC; letter-spacing:0.18em; text-transform:uppercase; text-align:center;
            margin-bottom: 1rem;
        }
        section[data-testid="stSidebar"] .sm-side-caption {
            font-size: 0.78rem; color: #B7B3CC !important; text-align:center; line-height:1.5;
        }
        section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div,
        section[data-testid="stSidebar"] [data-baseweb="select"] {
            background: var(--sm-light) !important; border-radius: 12px !important;
        }
        section[data-testid="stSidebar"] [data-testid="stSelectbox"] label { color: #D8D5E8 !important; font-size:0.8rem; }

        section[data-testid="stSidebar"] [data-testid="stButton"] button {
            border-radius: 12px !important; text-align:left !important; justify-content:flex-start !important;
            font-family:'Plus Jakarta Sans',sans-serif; font-weight:600; margin-bottom: 4px;
            transition: transform .15s ease, background .2s ease;
        }
        section[data-testid="stSidebar"] button[kind="secondary"] {
            background: transparent !important; color: #C9C6DC !important; box-shadow:none !important;
            border: 1px solid transparent !important;
        }
        section[data-testid="stSidebar"] button[kind="secondary"]:hover {
            background: rgba(255,255,255,0.08) !important; transform: translateX(3px); color:#fff !important;
        }
        section[data-testid="stSidebar"] button[kind="primary"] {
            background: linear-gradient(120deg, var(--sm-accent), var(--sm-accent-soft)) !important; color:#fff !important;
            box-shadow: 0 6px 16px -6px rgba(115,106,134,0.6) !important;
        }

        /* ---------------- Shared cards / containers ---------------- */
        .sm-card {
            background: var(--sm-glass); border: 1px solid var(--sm-glass-border);
            backdrop-filter: blur(14px); border-radius: var(--sm-radius);
            padding: 1.25rem 1.5rem; margin-bottom: 1rem;
            box-shadow: 0 10px 26px -16px rgba(39,42,59,0.3);
            animation: smFadeInUp 0.45s ease both;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .sm-card:hover { transform: translateY(-3px); box-shadow: 0 16px 32px -14px rgba(115,106,134,0.4); }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--sm-glass) !important; border: 1px solid var(--sm-glass-border) !important;
            backdrop-filter: blur(14px); border-radius: var(--sm-radius) !important;
            transition: transform 0.22s ease, box-shadow 0.22s ease, border-color .22s ease;
            animation: smFadeInUp 0.45s ease both;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-5px); box-shadow: 0 18px 32px -16px rgba(115,106,134,0.45);
            border-color: var(--sm-accent) !important;
        }

        .sm-welcome {
            background: linear-gradient(120deg, var(--sm-accent), var(--sm-ink));
            border-radius: 24px; padding: 2.2rem 2.2rem; color:#fff; margin-bottom: 1.6rem; text-aign=right;
            animation: smFadeInUp 0.5s ease both; box-shadow: 0 22px 46px -24px rgba(39,42,59,0.5);
        }
        .sm-welcome h2 { margin:0; color:#fff; font-size: 1.8rem; letter-spacing:-0.01em; }
        .sm-welcome p { margin: 0.4rem 0 0 0; color: rgba(255,255,255,0.88); font-size:1.02rem; }

        .sm-qa-icon { font-size:1.7rem; animation: smFloat 4s ease-in-out infinite; }
        .sm-qa-title { font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; color: var(--sm-ink); margin: 0.4rem 0 0.15rem 0; font-size:1.02rem; }
        .sm-qa-desc { color: var(--sm-text-muted); font-size:1rem; margin-bottom: 0.7rem; line-height:1.5; }

        .sm-empty {
            text-align:center; padding: 3rem 1rem; border-radius: var(--sm-radius);
            background: var(--sm-glass); border: 1px dashed var(--sm-accent); animation: smFadeIn 0.5s ease both;
        }
        .sm-empty-emoji { font-size: 2.6rem; display:block; margin-bottom: 0.6rem; animation: smFloat 3.2s ease-in-out infinite; }
        .sm-empty-msg { color: var(--sm-text); font-size:1.02rem; font-weight:600; margin-bottom:0.3rem; }
        .sm-empty-hint { color: var(--sm-text-muted); font-size:0.88rem; line-height:1.5; }

        /* Metric cards */
        .sm-metric {
            background: transparent; border: 1px solid transparent; backdrop-filter: blur(14px);
            border-radius: 16px; padding: 1.1rem 1.2rem; text-align:center;
            box-shadow: 0 10px 24px -16px rgba(39,42,59,0.28);
            animation: smPop 0.4s ease both; transition: transform .2s ease, box-shadow .2s ease;
        }
        .sm-metric:hover { transform: translateY(-3px); box-shadow: 0 14px 28px -14px rgba(115,106,134,0.4); }
        .sm-metric .m-ico { font-size:1.3rem; margin-bottom:0.25rem; }
        .sm-metric .m-num { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.7rem; color: var(--sm-ink); line-height:1; }
        .sm-metric .m-lbl { color: "#DEDEE2"; font-size:0.78rem; margin-top:0.3rem; letter-spacing:0.02em; }

        /* Source citation chip */
        .sm-src {
            background: rgba(255,255,255,0.55); border:1px solid var(--sm-glass-border); border-radius:12px;
            padding: 0.6rem 0.8rem; margin-bottom:0.5rem; animation: smFadeInUp 0.35s ease both;
        }
        .sm-src .s-head {
            font-family:'Plus Jakarta Sans',sans-serif;
            font-weight:600; color: var(--sm-ink); font-size:14px;
            line-height:1.35;
        }
        .sm-src .s-snip {
            color: var(--sm-text-muted); font-size:0.8rem; line-height:1.45;
            margin-top:0.2rem;
            white-space: pre-wrap;
            word-break: break-word;
            overflow-wrap: anywhere;
            max-height: 4.8em;
            overflow: hidden;
        }
        .sm-src .s-snip * {
            font-size: inherit !important;
            line-height: inherit !important;
            font-weight: inherit !important;
            color: inherit !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        .sm-src pre {
            margin: 0 !important;
            font-family: inherit !important;
            font-size: 12px !important;
            line-height: 1.35 !important;
            white-space: pre-wrap !important;
            word-break: break-word !important;
            overflow-wrap: anywhere !important;
        }

        /* Notes output */
        .sm-note-line {
            font-family: 'Inter', sans-serif;
            font-size: 0.98rem;
            font-weight: 500;
            color: var(--sm-text);
            line-height: 1.6;
            margin: 0.18rem 0;
        }
        .sm-note-line--title {
            margin-top: 0;
            margin-bottom: 0.8rem;
        }
        .sm-note-line--section {
            margin-top: 0.95rem;
        }
        .sm-note-line--takeaways {
            margin-top: 1rem;
        }

        /* ---------------- Segmented control ---------------- */
        .sm-segmented div[role="radiogroup"] {
            display:flex; justify-content:center; gap:6px; background: var(--sm-glass);
            border: 1px solid var(--sm-glass-border); border-radius: 999px; padding: 5px; width:fit-content;
            margin: 0 auto 1.4rem auto; backdrop-filter: blur(10px);
        }
        
        .sm-segmented div[role="radiogroup"] label {
            background: transparent !important; border: none !important; border-radius: 999px !important;
            padding: 0.5rem 1.5rem !important; margin:0 !important; font-weight:600;
        }
        .sm-segmented div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(120deg, var(--sm-accent), var(--sm-ink)) !important; color:#fff !important;
        }
        .sm-segmented div[role="radiogroup"] label div:first-child { display:none; }

        /* Buttons (global) */
        .stButton > button, .stDownloadButton > button {
            background: linear-gradient(120deg, var(--sm-accent), var(--sm-ink));
            color: #fff; border: none; border-radius: 12px; padding: 0.58rem 1.3rem;
            font-weight: 600; font-family:'Plus Jakarta Sans',sans-serif;
            transition: transform 0.18s ease, box-shadow 0.18s ease, filter .18s ease;
            box-shadow: 0 6px 16px -6px rgba(39,42,59,0.45);
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-2px) scale(1.015); box-shadow: 0 10px 22px -6px rgba(115,106,134,0.55); color:#fff; filter: brightness(1.05);
        }
        .stButton > button:active { transform: translateY(0) scale(0.98); }
        .stButton > button:disabled { background: var(--sm-mist); color:#7a7886; box-shadow:none; }

        [data-testid="stFileUploaderDropzone"] {
            border-radius: 14px; border: 1.5px dashed var(--sm-accent); background: rgba(255,255,255,0.4);
            transition: border-color .2s ease, background .2s ease;
        }
        [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--sm-ink); background: rgba(255,255,255,0.55); }

        [data-testid="stExpander"] {
            background: var(--sm-glass); border: 1px solid var(--sm-glass-border); border-radius: 14px !important;
            backdrop-filter: blur(10px); margin-bottom:0.6rem; transition: transform .2s ease, box-shadow .2s ease;
            animation: smFadeInUp 0.4s ease both;
        }
        [data-testid="stExpander"]:hover { transform: translateY(-1px); box-shadow: 0 10px 22px -14px rgba(115,106,134,0.4); }

        # /* ---------------- Radio options: transparent by default, grayish theme when selected ---------------- */
        # div[role="radiogroup"] label {
        #     background: transparent; border: 1px solid var(--sm-glass-border); border-radius: 10px;
        #     padding: 0.5rem 0.85rem; margin-bottom: 6px; transition: background .18s ease, border-color .18s ease;
        # }
        
        # div[role="radiogroup"] label:hover { background: rgba(77,77,89,0.14); border-color: var(--sm-select); }
        # div[role="radiogroup"] label:has(input:checked) { background: var(--sm-select-soft); border-color: var(--sm-select); }

        div[role="radiogroup"] label {
        background: transparent; border-radius: 10px;
        padding: 0.5rem 0.85rem; margin-bottom: 6px; transition: background .18s ease;
        }
        div[role="radiogroup"] label:hover { background: rgba(77,77,89,0.14); }
        div[role="radiogroup"] label:has(input:checked) { background: var(--sm-select-soft); }

        /* Kill the default fieldset border Streamlit renders around radio groups (was showing as a stray line after each question) */
        [data-testid="stRadio"] fieldset { border: none !important; padding: 0 !important; margin: 0 !important; }
        [data-testid="stRadio"] > div { border: none !important; }

        /* Theme-consistent form controls (replace Streamlit's default red accents) */
        input[type="radio"], input[type="checkbox"] { accent-color: var(--sm-select) !important; }
        div[data-baseweb="slider"] div[role="slider"] {
            background-color: var(--sm-select) !important; border-color: var(--sm-select) !important;
            box-shadow: 0 0 0 2px rgba(77,77,89,0.25) !important;
        }
        div[data-baseweb="slider"] > div > div:nth-child(2) { background: var(--sm-select) !important; }
        span[data-baseweb="tag"] { background-color: var(--sm-select) !important; }

        [data-testid="stAlert"] { border-radius: 14px; animation: smFadeInUp 0.4s ease both; backdrop-filter: blur(6px); }
        [data-testid="stChatMessage"] { border-radius: 16px; animation: smFadeInUp 0.35s ease both; }

        /* ---------------- Chat input area: match the app background/theme instead of default white ---------------- */
        div[data-testid="stBottom"],
        div[data-testid="stBottom"] > div,
        div[data-testid="stBottomBlockContainer"] {
            background: linear-gradient(180deg, var(--sm-mist) 0%, var(--sm-accent) 100%) !important;
        }
        div[data-testid="stChatInput"] {
            background: var(--sm-glass) !important;
            border: 1px solid var(--sm-glass-border) !important;
            border-radius: 16px !important;
            box-shadow: 0 10px 26px -16px rgba(39,42,59,0.3);
        }
        div[data-testid="stChatInput"] textarea {
            background: transparent !important;
            color: var(--sm-text) !important;
        }
        div[data-testid="stChatInput"] textarea::placeholder { color: var(--sm-text-muted) !important; }
        div[data-testid="stChatInput"] button { background: transparent !important; box-shadow: none !important; }

        /* Chat typing indicator */
        .sm-typing { display:flex; gap:5px; align-items:center; padding: 0.3rem 0; }
        .sm-typing span { width:8px; height:8px; border-radius:50%; background: var(--sm-accent); animation: smBounce 1.2s infinite ease-in-out; }
        .sm-typing span:nth-child(2) { animation-delay: .10s; }
        .sm-typing span:nth-child(3) { animation-delay: .3s; }

        /* Flashcard flip */
        .sm-flash-wrap { display:flex; justify-content:center; margin: 1rem 0 0.6rem 0; perspective: 1400px; }
        .sm-flash-card {
            width: min(580px, 92%); height: 310px; position: relative; cursor:pointer;
            transform-style: preserve-3d; transition: transform 0.6s cubic-bezier(.4,.2,.2,1);
        }
        .sm-flash-card.flipped { transform: rotateY(180deg); }
        .sm-flash-face {
            position:absolute; inset:0; backface-visibility:hidden; border-radius: 22px;
            display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center;
            padding: 2rem; box-shadow: 0 20px 40px -18px rgba(39,42,59,0.4);
        }
        .sm-flash-front { background: linear-gradient(150deg, var(--sm-light), var(--sm-mist)); border: 1px solid var(--sm-glass-border); }
        .sm-flash-back { background: linear-gradient(150deg, var(--sm-accent), var(--sm-ink)); color: #fff; transform: rotateY(180deg); }
        .sm-flash-eyebrow { font-size:0.74rem; letter-spacing:0.1em; text-transform:uppercase; opacity:0.7; margin-bottom:0.7rem; font-weight:600; }
        .sm-flash-face .sm-flash-text { font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:1.25rem; color: var(--sm-ink); line-height:1.4; }
        .sm-flash-back .sm-flash-text { color:#fff; }
        .sm-flash-progress { text-align:center; color: var(--sm-text-muted); font-size:0.85rem; margin: 0.3rem 0 1rem 0; font-weight:500; }
        .sm-flash-tap-hint { text-align:center; color: var(--sm-accent); font-size:0.78rem; margin-bottom: 0.6rem; font-weight:500; }
        .sm-flash-flip-row { display:flex; justify-content:center; margin: 0.25rem 0 0.7rem 0; }
        .sm-flash-flip-row .stButton { width: auto; }
        .sm-flash-flip-row .stButton > button { width: auto; min-width: 8.5rem; padding: 0.45rem 1rem; border-radius: 999px; }
        .sm-flash-actions .stButton > button { min-height: 2.7rem; }

        # /* Invisible button overlay placed directly on top of the flashcard so a click on the card flips it */
        # div[data-testid="stVerticalBlock"] > div[data-testid="element-container"]:has(.sm-flash-wrap) + div[data-testid="element-container"] {
        #     margin-top: -336px; position: relative; z-index: 5;
        # }
        # div[data-testid="stVerticalBlock"] > div[data-testid="element-container"]:has(.sm-flash-wrap) + div[data-testid="element-container"] .stButton button {
        #     width: min(580px, 92%); height: 310px; margin: 0 auto; display:block;
        #     background: transparent !important; border: none !important; box-shadow: none !important; color: transparent !important;
        # }
        # div[data-testid="stVerticalBlock"] > div[data-testid="element-container"]:has(.sm-flash-wrap) + div[data-testid="element-container"] .stButton button:hover {
        #     background: transparent !important; transform: none !important;
        # }

        .sm-flip-btn-row { display:flex; justify-content:center; margin: 0 0 0.8rem 0; }
        .sm-flip-btn-row .stButton { width: min(150px, 30%); }
        .sm-flip-btn-row .stButton > button { width: 10%; }

        /* Score banner */
        .sm-score {
            text-align:center; border-radius: 18px; padding: 1.6rem 1rem; margin-bottom:1rem;
            background: linear-gradient(120deg, var(--sm-accent), var(--sm-ink)); color:#fff;
            animation: smPop 0.5s ease both; box-shadow: 0 18px 38px -18px rgba(39,42,59,0.5);
        }
        .sm-score .s-emoji { font-size:2rem; display:block; margin-bottom:0.3rem; }
        .sm-score .s-value { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:2rem; }
        .sm-score .s-sub { color: rgba(255,255,255,0.85); font-size:0.9rem; margin-top:0.2rem; }

        /* Review item */
        .sm-review { border-radius: 14px; padding: 0.9rem 1.1rem; margin-bottom:0.6rem; animation: smFadeInUp 0.35s ease both; }
        .sm-review.ok { background: var(--sm-success-soft); border-left: 4px solid var(--sm-success); }
        .sm-review.no { background: var(--sm-error-soft); border-left: 4px solid var(--sm-error); }
        .sm-review .r-q { font-weight:700; color: var(--sm-ink); margin-bottom:0.25rem; }
        .sm-review .r-line { color: var(--sm-text-muted); font-size:0.83rem; line-height:1.5; }
        .sm-review .r-line strong { color: var(--sm-ink); }

        /* Document pipeline animation */
        .sm-pipeline { display:flex; align-items:center; justify-content:space-between; margin: 1.2rem 0 1.6rem 0; position:relative; }
        .sm-pipeline::before {
            content:""; position:absolute; top:26px; left:6%; right:6%; height:2px; z-index:0;
            background: repeating-linear-gradient(90deg, var(--sm-accent) 0 10px, transparent 10px 18px);
            animation: smDash 1.4s linear infinite;
        }
        .sm-pipe-step { position:relative; z-index:1; text-align:center; flex:1; animation: smPop 0.5s ease both; }
        .sm-pipe-icon {
            width:52px; height:52px; border-radius:50%; margin:0 auto 0.5rem auto; display:flex; align-items:center; justify-content:center;
            font-size:1.3rem; background: linear-gradient(135deg, var(--sm-accent), var(--sm-ink)); color:#fff;
            border: 3px solid var(--sm-light); box-shadow: 0 8px 18px -8px rgba(39,42,59,0.5);
        }
        .sm-pipe-label { font-size:0.82rem; color: var(--sm-ink); font-weight:600; }

        /* Settings tile */
        .sm-tile {
            text-align:center; border-radius: 16px; padding: 1.4rem 1rem;
            background: var(--sm-glass); border: 1px solid var(--sm-glass-border); backdrop-filter: blur(14px);
            animation: smFadeInUp 0.45s ease both; transition: transform .2s ease;
        }
        .sm-tile:hover { transform: translateY(-3px); }
        .sm-tile .t-ico { font-size:1.5rem; margin-bottom:0.4rem; }
        .sm-tile .t-title { font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; color: var(--sm-ink); }
        .sm-tile .t-sub { color: var(--sm-text-muted); font-size:0.85rem; margin-top:0.3rem; line-height:1.5; }

        .sm-chip {
            display:inline-block; padding:0.28rem 0.75rem; border-radius:999px; font-size:0.72rem; font-weight:600;
            background: rgba(46,133,87,0.16); color: var(--sm-success); letter-spacing:0.02em;
        }

        ::-webkit-scrollbar { width: 9px; height: 9px; }
        ::-webkit-scrollbar-thumb { background: linear-gradient(var(--sm-accent), var(--sm-ink)); border-radius: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_title(emoji: str, title: str, subtitle: str = ""):
    parts = title.split()
    head = " ".join(parts[:-1]) if len(parts) > 1 else title
    tail = f' <span class="accent">{parts[-1]}</span>' if len(parts) > 1 else ""
    st.markdown(f'<div class="sm-section-title">{emoji} {head}{tail}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sm-title-rule"></div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="sm-section-sub">{subtitle}</div>', unsafe_allow_html=True)


def empty_state(emoji: str, message: str, hint: str = ""):
    hint_html = f'<div class="sm-empty-hint">{hint}</div>' if hint else ""
    st.markdown(
        f'<div class="sm-empty"><span class="sm-empty-emoji">{emoji}</span>'
        f'<div class="sm-empty-msg">{message}</div>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def metric_card(icon: str, value, label: str):
    st.markdown(
        f'<div class="sm-metric"><div class="m-ico">{icon}</div>'
        f'<div class="m-num">{value}</div><div class="m-lbl">{label}</div></div>',
        unsafe_allow_html=True,
    )


def greeting() -> str:
    h = _dt.datetime.now().hour
    if h < 12:
        return "Good morning"
    if h < 18:
        return "Good afternoon"
    return "Good evening"


inject_design_system()


# ============================================================
# Session state initialization
# ============================================================

def init_state():
    defaults = {
        "chunks": [],
        "processed_files": [],
        "vector_store_service": None,
        "retriever": None,
        "rag_chain": None,
        "chat_history": [],
        "notes": {},
        "flashcards": None,
        "quiz": None,
        "quiz_answers": {},
        "quiz_submitted": False,
        "exam_active": False,
        "exam_start_time": None,
        "exam_quiz": None,
        "exam_answers": {},
        "exam_submitted": False,
        "exam_minutes": 10,
        "view": "landing",
        "nav": "Home",
        "flash_index": 0,
        "flash_flipped": False,
        "flash_order": [],
        "questions_asked": 0,
        "exams_created": 0,
        "just_processed": False,
        "exam_mode_choice": "📝 Quiz Generator",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


@st.cache_resource(show_spinner=False)
def get_rag_chain():
    return build_rag_chain(llm)


@st.cache_resource(show_spinner=False)
def get_notes_chain():
    return build_notes_chain(generative_llm)


@st.cache_resource(show_spinner=False)
def get_flashcard_chain():
    return build_flashcard_chain(generative_llm)


@st.cache_resource(show_spinner=False)
def get_quiz_chain():
    return build_quiz_chain(generative_llm)


def require_documents() -> bool:
    if not st.session_state.chunks:
        empty_state(
            "📄",
            "No documents loaded yet",
            "Head to the <strong>Documents</strong> page and upload a PDF, TXT, or DOCX to unlock this tool.",
        )
        return False
    return True


def process_uploaded_files(uploaded_files) -> None:
    Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
    saved_paths = []
    for uploaded_file in uploaded_files:
        save_path = Path(UPLOAD_FOLDER) / uploaded_file.name
        save_path.write_bytes(uploaded_file.getvalue())
        saved_paths.append(str(save_path))

    with st.spinner("Loading, cleaning, and chunking documents..."):
        chunks = process_documents(saved_paths, validate=True)

    with st.spinner("Embedding chunks and building the vector store..."):
        vs_service = VectorStoreService()
        try:
            vs_service.create(chunks)
        except Exception as exc:
            # Streamlit Cloud's default uncaught-exception screen redacts
            # the real message ("to prevent data leaks") and only writes
            # it to the app's server-side logs, which are easy to miss.
            # Catching here and rendering with st.exception() shows the
            # actual error (e.g. the Hugging Face API's real HTTP status
            # and response body) directly in the UI instead.
            st.error("Embedding failed -- see the full error below.")
            st.exception(exc)
            st.stop()

    st.session_state.chunks = chunks
    st.session_state.processed_files = [Path(p).name for p in saved_paths]
    st.session_state.vector_store_service = vs_service

    retriever_service = RetrieverService(vs_service.vectorstore, documents=vs_service.documents)
    # Similarity search is noticeably faster than hybrid retrieval and is
    # a good default for interactive chat when you want response speed.
    st.session_state.retriever = retriever_service.get_retriever(search_type="similarity")
    st.session_state.rag_chain = get_rag_chain()

    st.session_state.notes = None
    st.session_state.flashcards = None
    st.session_state.quiz = None
    st.session_state.chat_history = []
    st.session_state.just_processed = True

    st.toast(f"Processed {len(saved_paths)} file(s) into {len(chunks)} chunks.", icon="✅")


def render_pipeline_animation():
    steps = [("📄", "Uploaded"), ("🧹", "Cleaned & Chunked"), ("🧠", "Embedded"), ("🗂️", "Indexed"), ("✅", "Ready")]
    html = '<div class="sm-pipeline">'
    for icon, label in steps:
        html += f'<div class="sm-pipe-step"><div class="sm-pipe-icon">{icon}</div><div class="sm-pipe-label">{label}</div></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# LANDING PAGE
# ============================================================

def render_landing():
    st.markdown(
        """
        <div class="sm-hero-land">
            <div class="sm-blob" style="width:230px; height:230px; top:-44px; left:-44px;"></div>
            <div class="sm-blob" style="width:170px; height:170px; bottom:-34px; right:10%; animation-delay:5s;"></div>
            <div class="sm-eyebrow"><h2 class="grad">💡STUDYMATE </h2></div>
            <h1>Your AI Study Companion,<br><span class="grad">powered by RAG + LLM intelligence</span></h1>
            <p>Chat with your notes, generate flashcards, create exams, and learn smarter — all grounded in your own documents.</p>
            <div class="sm-hero-stats">
                <div><div class="hs-num">4</div><div class="hs-lbl">Study tools</div></div>
                <div><div class="hs-num">PDF · TXT · DOCX</div><div class="hs-lbl">Supported formats</div></div>
                <div><div class="hs-num">100%</div><div class="hs-lbl">Cited answers</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    gs1, gs2, gs3 = st.columns([1.4, 1, 1.4])
    with gs2:
        if st.button("Get Started →", use_container_width=True, key="hero_get_started"):
            st.session_state.view = "dashboard"
            st.session_state.nav = "Home"
            st.rerun()

    section_title( "From raw documents", "to an exam-ready study kit, in four steps.")
    st.markdown('<div class="sm-how-wrap"><div class="sm-how-line"></div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    steps = [
        ("1", "Upload Documents", "Drop in PDFs, text, or Word files."),
        ("2", "Process & Embed", "We chunk and index everything automatically."),
        ("3", "Ask Questions", "Chat and get cited, grounded answers."),
        ("4", "Generate Material", "Notes, flashcards, quizzes, and exams."),
    ]
    for col, (num, title, desc) in zip([s1, s2, s3, s4], steps):
        with col:
            st.markdown(
                f"""<div class="sm-step"><div class="sm-step-num">{num}</div>
                <strong>{title}</strong><p>{desc}</p></div>""",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# DASHBOARD PAGES
# ============================================================

def render_dashboard_home():
    st.markdown(
        f"""
        <div class="sm-welcome">
            <h2>{greeting()}✨</h2>
            <p>Ready to learn smarter today?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title("⚡", "Quick actions", "Jump straight into what you need.")
    quick_actions = [
        ("📁", "Upload Document", "Add PDFs, TXT, or DOCX to power every tool below.", "Documents"),
        ("💬", "Start Chat", "Ask questions and get cited, grounded answers.", "Chat"),
        ("📝", "Generate Notes", "Turn your material into structured notes.", "Notes"),
        ("🎯", "Create Exam", "Build a timed practice exam from your docs.", "Exam"),
    ]
    cols = st.columns(4)
    for col, (icon, title, desc, target) in zip(cols, quick_actions):
        with col:
            with st.container(border=True):
                st.markdown(
                    f"""<div style="text-align:center;">
                        <div class="sm-qa-icon">{icon}</div>
                        <div class="sm-qa-title">{title}</div>
                        <div class="sm-qa-desc">{desc}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if st.button("Open →", use_container_width=True, key=f"qa_{target}"):
                    st.session_state.nav = target
                    st.rerun()

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    section_title( "Your study", "activity for this session.")
    n_cards = len(st.session_state.flashcards.flashcards) if st.session_state.flashcards else 0
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("📁", len(st.session_state.processed_files), "Documents uploaded")
    with m2:
        metric_card("💬", st.session_state.questions_asked, "Questions asked")
    with m3:
        metric_card("🧠", n_cards, "Flashcards generated")
    with m4:
        metric_card("🎯", st.session_state.exams_created, "Exams created")


def render_documents_page():
    section_title("📁", "Your documents", "Upload PDFs, TXT, or DOCX files to power every tool in the sidebar.")

    uploaded_files = st.file_uploader(
        "Drop your files here", type=["pdf", "txt", "docx"], accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if st.button("🚀 Process documents", disabled=not uploaded_files):
        process_uploaded_files(uploaded_files)
        st.rerun()

    if st.session_state.processed_files:
        if st.session_state.just_processed:
            render_pipeline_animation()
            st.session_state.just_processed = False

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        table_count = sum(1 for c in st.session_state.chunks if c.metadata.get("content_type") == "table")
        st.caption(f"{len(st.session_state.chunks)} chunks indexed · {table_count} tables kept intact")
        for name in st.session_state.processed_files:
            st.markdown(
                f"""<div class="sm-card" style="display:flex; justify-content:space-between; align-items:center;">
                    <div>📄 <strong>{name}</strong></div>
                    <span class="sm-chip">Indexed</span>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        empty_state(
            "🗂️",
            "No documents yet",
            "Upload one or more files above and hit <strong>Process documents</strong> to get started.",
        )


def render_chat_page():
    section_title("💬", "Chat assistant", "Source-aware answers, grounded in your own documents.")
    if not require_documents():
        return

    # for entry in st.session_state.chat_history:
    #     with st.chat_message(entry["role"]):
    #         st.markdown(entry["content"])
    #         if entry.get("sources"):
    #             with st.expander(f"🔎 Sources · {len(entry['sources'])} cited"):
    #                 for src in entry["sources"]:
    #                     page_label = f" · page {src['page']}" if src.get("page") not in (None, 0) else ""
    #                     st.markdown(
    #                         f"""<div class="sm-src">
    #                             <div class="s-head">📄 {src['source']}{page_label}</div>
    #                             <div class="s-snip">{src['snippet']}</div>
    #                         </div>""",
    #                         unsafe_allow_html=True,
    #                     )
    #                     #page_label = f" · page {src['page']}" if src.get("page") not in (None, 0) else ""

    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

            if entry.get("sources"):
                top_sources = entry["sources"][:3]

                with st.expander(f"🔎 Sources · Top {len(top_sources)} cited"):
                    for src in top_sources:
                        page_label = (
                            f" · page {src['page']}"
                            if src.get("page") not in (None, 0)
                            else ""
                        )
                        source_name = html.escape(str(src.get("source", "unknown")))
                        snippet = html.escape(str(src.get("snippet", "")))

                        st.markdown(
                            f"""
                            <div class="sm-src">
                                <div class="s-head">
                                    📄 {source_name}{page_label}
                                </div>
                                <pre class="s-snip">{snippet}</pre>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


    question = st.chat_input("Ask anything about your documents...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.questions_asked += 1
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            typing_placeholder = st.empty()
            typing_placeholder.markdown(
                '<div class="sm-typing"><span></span><span></span><span></span></div>',
                unsafe_allow_html=True,
            )
            retrieved_docs = st.session_state.retriever.invoke(question)
            answer = st.session_state.rag_chain.invoke({"question": question, "context": retrieved_docs})
            typing_placeholder.markdown(answer)

        sources = [
            {
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page"),
                "snippet": d.page_content[:200] + "...",
            }
            for d in retrieved_docs
        ]
 
        
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
        st.rerun()


def render_notes_page():
    section_title("📝", "Notes generator", "Turn your documents into structured notes or topic summaries.")
    if not require_documents():
        return

    doc_options = st.session_state.processed_files
    selected_docs = st.multiselect(
        "Generate notes from", doc_options, default=doc_options,
        help="Pick one or more indexed documents. Leave all selected to use everything.",
    )

    if len(doc_options) > 1 and len(selected_docs) == len(doc_options):
        st.info("Generating notes for every document can take a while. For faster results, pick one document at a time.")

    style = st.selectbox(
        "Style", NOTES_STYLES, index=NOTES_STYLES.index(DEFAULT_NOTES_STYLE),
        help="'topic_summary' is the Topic Summaries feature -- same generator, different output shape.",
    )

    if st.button("✨ Generate notes", disabled=not selected_docs):
        with st.spinner("Generating notes..."):
            notes_chain = get_notes_chain()
            notes_by_doc = []

            for doc_name in selected_docs:
                target_chunks = [
                    c for c in st.session_state.chunks
                    if Path(str(c.metadata.get("source", ""))).name == doc_name
                ]

                if not target_chunks:
                    continue

                notes = notes_chain(target_chunks, style=style, document_name=doc_name)
                notes_by_doc.append((doc_name, notes))

            st.session_state.notes = notes_by_doc

    if st.session_state.notes:
        notes_items = st.session_state.notes

        # Backward compatibility: if an older session stored a single NotesOutput,
        # render it as one document instead of breaking the page.
        if hasattr(notes_items, "title"):
            notes_items = [("Selected document", notes_items)]

        for doc_name, notes in notes_items:
            st.markdown('<div class="sm-card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="sm-note-line sm-note-line--title">{doc_name}: {notes.title}</div>',
                unsafe_allow_html=True,
            )
            for section in notes.sections:
                st.markdown(
                    f'<div class="sm-note-line sm-note-line--section">{section.heading}</div>',
                    unsafe_allow_html=True,
                )
                for point in section.bullet_points:
                    st.markdown(
                        f'<div class="sm-note-line">• {point}</div>',
                        unsafe_allow_html=True,
                    )
            st.markdown('<div class="sm-note-line sm-note-line--takeaways">🔑 Key Takeaways</div>', unsafe_allow_html=True)
            for takeaway in notes.key_takeaways:
                st.markdown(
                    f'<div class="sm-note-line">• {takeaway}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)


def render_flashcards_page():
    section_title("🧠", "Flashcards", "Active recall One card at a time.")
    if not require_documents():
        return

    def move_flash_index(step: int) -> None:
        total = len(st.session_state.flashcards.flashcards) if st.session_state.flashcards else 0
        if total <= 0:
            return
        st.session_state.flash_index = max(0, min(st.session_state.flash_index + step, total - 1))
        st.session_state.flash_flipped = False

    def toggle_flash_flip() -> None:
        st.session_state.flash_flipped = not st.session_state.flash_flipped

    def shuffle_flash_order() -> None:
        if not st.session_state.flashcards:
            return
        st.session_state.flash_order = list(range(len(st.session_state.flashcards.flashcards)))
        random.shuffle(st.session_state.flash_order)
        st.session_state.flash_index = 0
        st.session_state.flash_flipped = False

    num_flashcards = st.slider("Number of flashcards", 5, 20, DEFAULT_FLASHCARD_COUNT)
    if st.button("🃏 Generate flashcards"):
        with st.spinner("Generating flashcards..."):
            flashcard_chain = get_flashcard_chain()
            st.session_state.flashcards = flashcard_chain(st.session_state.chunks, num_flashcards=num_flashcards)
        st.session_state.flash_index = 0
        st.session_state.flash_flipped = False
        st.session_state.flash_order = list(range(len(st.session_state.flashcards.flashcards)))

    if not st.session_state.flashcards:
        return

    cards = st.session_state.flashcards.flashcards
    if len(st.session_state.flash_order) != len(cards):
        st.session_state.flash_order = list(range(len(cards)))
    if st.session_state.flash_index >= len(cards):
        st.session_state.flash_index = 0

    idx = st.session_state.flash_order[st.session_state.flash_index]
    card = cards[idx]
    flip_class = "flipped" if st.session_state.flash_flipped else ""

    st.markdown('<div class="sm-flash-tap-hint">👆 Click the card to flip it</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sm-flash-wrap">
            <div class="sm-flash-card {flip_class}">
                <div class="sm-flash-face sm-flash-front">
                    <div class="sm-flash-eyebrow">Question · {card.topic}</div>
                    <div class="sm-flash-text">{card.question}</div>
                </div>
                <div class="sm-flash-face sm-flash-back">
                    <div class="sm-flash-eyebrow">Answer</div>
                    <div class="sm-flash-text">{card.answer}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sm-flip-btn-row">', unsafe_allow_html=True)
    flip_left, flip_mid, flip_right = st.columns([1, 0.65, 1])
    with flip_mid:
        st.button("🔄 Flip", key="flash_flip_btn", use_container_width=True, on_click=toggle_flash_flip)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="sm-flash-progress">Card {st.session_state.flash_index + 1} of {len(cards)}</div>',
        unsafe_allow_html=True,
    )

    b1, b2, b3 = st.columns(3)
    with b1:
        st.button(
            "⬅️ Previous",
            key="flash_prev_btn",
            use_container_width=True,
            disabled=st.session_state.flash_index <= 0,
            on_click=move_flash_index,
            args=(-1,),
        )
    with b2:
        st.button("🔀 Shuffle", key="flash_shuffle_btn", use_container_width=True, on_click=shuffle_flash_order)
    with b3:
        st.button(
            "Next ➡️",
            key="flash_next_btn",
            use_container_width=True,
            disabled=st.session_state.flash_index >= len(cards) - 1,
            on_click=move_flash_index,
            args=(1,),
        )


def render_exam_page():
    section_title("🎯", "Exam mode", "A timed run of the Quiz Generator, with a final scored review.")
    if not require_documents():
        return

    st.markdown('<div class="sm-segmented">', unsafe_allow_html=True)
    mode = st.radio(
        "Mode", ["📝 Quiz Generator", "🎯 Timed Exam"],
        index=["📝 Quiz Generator", "🎯 Timed Exam"].index(st.session_state.exam_mode_choice),
        horizontal=True, label_visibility="collapsed", key="exam_mode_radio",
    )
    st.session_state.exam_mode_choice = mode
    st.markdown('</div>', unsafe_allow_html=True)

    if mode == "📝 Quiz Generator":
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                num_questions = st.slider("Number of questions", 3, 15, DEFAULT_QUIZ_QUESTION_COUNT, key="quiz_num_q")
            with col2:
                difficulty = st.selectbox(
                    "Difficulty", QUIZ_DIFFICULTIES, index=QUIZ_DIFFICULTIES.index(DEFAULT_QUIZ_DIFFICULTY),
                    key="quiz_difficulty",
                )
            if st.button("📝 Generate quiz"):
                with st.spinner("Generating quiz..."):
                    quiz_chain = get_quiz_chain()
                    st.session_state.quiz = quiz_chain(
                        st.session_state.chunks, num_questions=num_questions, difficulty=difficulty,
                    )
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = False

        if st.session_state.quiz:
            questions = st.session_state.quiz.questions
            for i, q in enumerate(questions):
                st.markdown('<div class="sm-card">', unsafe_allow_html=True)
                st.markdown(f"**Q{i + 1}. {q.question}**")
                choice = st.radio(
                    "Select an answer", options=list(range(len(q.options))),
                    format_func=lambda idx, opts=q.options: opts[idx], key=f"quiz_q_{i}",
                    label_visibility="collapsed",
                )
                st.session_state.quiz_answers[i] = choice
                st.markdown('</div>', unsafe_allow_html=True)

            if st.button("✅ Submit quiz"):
                st.session_state.quiz_submitted = True

            if st.session_state.quiz_submitted:
                score = sum(1 for i, q in enumerate(questions)
                            if st.session_state.quiz_answers.get(i) == q.correct_answer_index)
                pct = int(round(score / max(1, len(questions)) * 100))
                emoji = "🎉" if pct >= 80 else ("👍" if pct >= 50 else "📚")
                st.markdown(
                    f"""<div class="sm-score"><span class="s-emoji">{emoji}</span>
                    <div class="s-value">Score: {score} / {len(questions)}</div>
                    <div class="s-sub">{pct}% correct · {len(questions) - score} to review</div></div>""",
                    unsafe_allow_html=True,
                )
                for i, q in enumerate(questions):
                    correct = st.session_state.quiz_answers.get(i) == q.correct_answer_index
                    cls = "ok" if correct else "no"
                    icon = "✅" if correct else "❌"
                    st.markdown(
                        f"""<div class="sm-review {cls}">
                            <div class="r-q">{icon} Q{i + 1}. {q.question}</div>
                            <div class="r-line">{q.explanation}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

    else:  # Timed Exam
        if not st.session_state.exam_active:
            with st.container(border=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    exam_num_questions = st.slider("Questions", 5, 20, 10, key="exam_num_q")
                with col2:
                    exam_difficulty = st.selectbox(
                        "Difficulty", QUIZ_DIFFICULTIES, index=QUIZ_DIFFICULTIES.index("medium"), key="exam_difficulty",
                    )
                with col3:
                    exam_minutes = st.number_input("Time limit (minutes)", 1, 60, 10)

                if st.button("🎯 Start exam"):
                    with st.spinner("Building your exam..."):
                        quiz_chain = get_quiz_chain()
                        st.session_state.exam_quiz = quiz_chain(
                            st.session_state.chunks, num_questions=exam_num_questions, difficulty=exam_difficulty,
                        )
                    st.session_state.exam_active = True
                    st.session_state.exam_start_time = time.time()
                    st.session_state.exam_minutes = exam_minutes
                    st.session_state.exam_answers = {}
                    st.session_state.exam_submitted = False
                    st.session_state.exams_created += 1
                    st.rerun()
        else:
            elapsed = time.time() - st.session_state.exam_start_time
            total_seconds = st.session_state.exam_minutes * 60
            remaining = max(0, total_seconds - elapsed)
            time_up = remaining <= 0
            mins, secs = divmod(int(remaining), 60)
            pct_left = remaining / total_seconds if total_seconds else 0
            timer_color = "#736A86" if pct_left > 0.25 else ("#C99042" if pct_left > 0.1 else "#C14545")

            st.markdown(
                f"""
                <div class="sm-card" style="text-align:center;">
                    <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:2.2rem; color:{timer_color}; letter-spacing:0.04em;">
                        ⏱️ {mins:02d}:{secs:02d}
                    </div>
                    <div style="background:rgba(115,106,134,0.18); border-radius:999px; height:8px; margin-top:0.7rem; overflow:hidden;">
                        <div style="width:{pct_left*100:.1f}%; height:100%; background:linear-gradient(90deg, var(--sm-accent), var(--sm-ink)); transition: width 1s linear;"></div>
                    </div>
                    <div style="color:var(--sm-text-muted); font-size:0.85rem; margin-top:0.5rem;">Time remaining</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if time_up and not st.session_state.exam_submitted:
                st.warning("⏰ Time's up! Submitting automatically.")
                st.session_state.exam_submitted = True

            questions = st.session_state.exam_quiz.questions

            if not st.session_state.exam_submitted:
                for i, q in enumerate(questions):
                    st.markdown('<div class="sm-card">', unsafe_allow_html=True)
                    st.markdown(f"**Q{i + 1}. {q.question}**")
                    choice = st.radio(
                        "Select an answer", options=list(range(len(q.options))),
                        format_func=lambda idx, opts=q.options: opts[idx], key=f"exam_q_{i}",
                        label_visibility="collapsed",
                    )
                    st.session_state.exam_answers[i] = choice
                    st.markdown('</div>', unsafe_allow_html=True)

                if st.button("✅ Submit exam"):
                    st.session_state.exam_submitted = True
                    st.rerun()

            if st.session_state.exam_submitted:
                score = sum(1 for i, q in enumerate(questions)
                            if st.session_state.exam_answers.get(i) == q.correct_answer_index)
                pct = int(round(score / max(1, len(questions)) * 100))
                emoji = "🎉" if pct >= 80 else ("👍" if pct >= 50 else "📚")
                st.markdown(
                    f"""<div class="sm-score"><span class="s-emoji">{emoji}</span>
                    <div class="s-value">Final score: {score} / {len(questions)}</div>
                    <div class="s-sub">{pct}% correct · {len(questions) - score} to review</div></div>""",
                    unsafe_allow_html=True,
                )
                for i, q in enumerate(questions):
                    user_choice = st.session_state.exam_answers.get(i)
                    correct = user_choice == q.correct_answer_index
                    cls = "ok" if correct else "no"
                    icon = "✅" if correct else "❌"
                    chosen_label = q.options[user_choice] if user_choice is not None else "(no answer)"
                    correct_label = q.options[q.correct_answer_index]
                    st.markdown(
                        f"""<div class="sm-review {cls}">
                            <div class="r-q">{icon} Q{i + 1}. {q.question}</div>
                            <div class="r-line"><strong>Your answer:</strong> {chosen_label}</div>
                            <div class="r-line"><strong>Correct answer:</strong> {correct_label}</div>
                            <div class="r-line" style="margin-top:0.3rem;">{q.explanation}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                if st.button("🔄 Start a new exam"):
                    st.session_state.exam_active = False
                    st.rerun()


def render_settings_page():
    section_title("⚙️", "Settings", "Model, storage, and version info for this session.")

    m1, m2 = st.columns(2)
    with m1:
        st.markdown(
            """<div class="sm-tile">
                <div class="t-ico">🧠</div><div class="t-title">AI Model</div>
                <div class="t-sub">Retrieval runs through your configured Hugging Face models (chat + generative chains).</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""<div class="sm-tile">
                <div class="t-ico">🗄️</div><div class="t-title">Storage</div>
                <div class="t-sub">{len(st.session_state.processed_files)} document(s) · {len(st.session_state.chunks)} chunks indexed in this session.</div>
            </div>""",
            unsafe_allow_html=True,
        )
    m3, m4 = st.columns(2)
    with m3:
        st.markdown(
            """<div class="sm-tile">
                <div class="t-ico">🔐</div><div class="t-title">Privacy</div>
                <div class="t-sub">Documents are processed locally in this session and never leave your machine.</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            """<div class="sm-tile">
                <div class="t-ico">🏷️</div><div class="t-title">Version</div>
                <div class="t-sub">StudyMate AI · session-local demo build</div>
            </div>""",
            unsafe_allow_html=True,
        )


# ============================================================
# DASHBOARD SHELL (sidebar nav + quick jump + routing)
# ============================================================

NAV_SECTIONS = ["Home", "Chat", "Notes", "Flashcards", "Exam", "Documents", "Settings"]
NAV_ICONS = {
    "Home": "🏠", "Chat": "💬", "Notes": "📝", "Flashcards": "🧠",
    "Exam": "🎯", "Documents": "📁", "Settings": "⚙️",
}


def render_dashboard():
    with st.sidebar:
        st.markdown('<div class="sm-side-logo">STUDYMATE</div>', unsafe_allow_html=True)
        st.markdown('<div class="sm-side-tag">AI study companion</div>', unsafe_allow_html=True)

        quick_jump = st.selectbox(
            "Quick jump", ["🔎 Search a section..."] + NAV_SECTIONS,
            label_visibility="collapsed",
            help="Stand-in for a Ctrl+K command palette -- Streamlit can't capture "
                 "global keyboard shortcuts without a custom JS component.",
        )
        if quick_jump in NAV_SECTIONS and quick_jump != st.session_state.nav:
            st.session_state.nav = quick_jump
            st.rerun()

        st.markdown("---")
        for section in NAV_SECTIONS:
            is_active = st.session_state.nav == section
            if st.button(
                f"{NAV_ICONS[section]}  {section}", key=f"nav_{section}",
                use_container_width=True, type="primary" if is_active else "secondary",
            ):
                st.session_state.nav = section
                st.rerun()

        st.markdown("---")
        if st.session_state.processed_files:
            st.markdown(
                f'<div class="sm-side-caption">📁 {len(st.session_state.processed_files)} document(s) indexed</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="sm-side-caption">📁 No documents yet</div>', unsafe_allow_html=True)

        if st.button("← Back to landing page", use_container_width=True):
            st.session_state.view = "landing"
            st.rerun()

    page = st.session_state.nav
    if page == "Home":
        render_dashboard_home()
    elif page == "Chat":
        render_chat_page()
    elif page == "Notes":
        render_notes_page()
    elif page == "Flashcards":
        render_flashcards_page()
    elif page == "Exam":
        render_exam_page()
    elif page == "Documents":
        render_documents_page()
    elif page == "Settings":
        render_settings_page()


# ============================================================
# Entry point
# ============================================================

if st.session_state.view == "landing":
    render_landing()
else:
    render_dashboard()