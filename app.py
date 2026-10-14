import os

# Entry point for Streamlit to run directly with: streamlit run app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
dashboard_app = os.path.join(current_dir, "dashboard", "app.py")

with open(dashboard_app, "r", encoding="utf-8") as f:
    code = compile(f.read(), dashboard_app, "exec")
    exec(code, globals())
