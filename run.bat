@echo off
REM Launch Socratix Streamlit UI (works even when streamlit is not on PATH)
cd /d "%~dp0"
python -m streamlit run app.py
