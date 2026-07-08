# scripts/generate_rag_dashboard.py
import pandas as pd
from src.observability.mlflow_rag_tracker import MLFlowRAGTracker
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

def generate_local_dashboard():
    tracker = MLFlowRAGTracker(tracking_uri="mlruns")
    df = tracker.get_latest_metrics_df()
    
    if df.empty:
        print("❌ No metrics logged yet. Run the app or evaluation first.")
        return

    fig = make_subplots(rows=2, cols=2, subplot_titles=("Query Latency Distribution", "Avg Retrieved Chunks", 
                                                         "Evaluation Faithfulness", "Request Count"))
    
    fig.add_trace(go.Box(y=df.get("query_latency_sec", []), name="HTTP/WS Latency", boxmean=True), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df.get("retrieved_chunks", df.get("avg_retrieved_chunks", [])), name="Chunks"), row=1, col=2)
    fig.add_trace(go.Histogram(x=df.filter(like="eval_faithfulness").iloc[:, -5:].values.flatten(), 
                               nbinsx=20, name="Faithfulness Scores"), row=2, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df.get("request_count", [len(df)]*len(df)), name="Requests"), row=2, col=2)
    
    fig.update_layout(height=600, title_text="🤖 RAG System Observability Dashboard (Local/MLflow)", showlegend=False)
    fig.write_html("rag_dashboard.html")
    print("✅ Dashboard generated: rag_dashboard.html")

if __name__ == "__main__":
    generate_local_dashboard()
