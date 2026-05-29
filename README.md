# Stock Screener

Streamlit stock screener app.

## Run locally

```powershell
.\venv\Scripts\python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, create a new app from that repository.
3. Set the entrypoint file to `app.py`.
4. Use Python 3.12 unless you have a reason to pin a different supported version.

The app dependencies are declared in `requirements.txt`. Do not commit `.streamlit/secrets.toml` if you add secrets later.
