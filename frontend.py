import gradio as gr
from backend import WealthManagementAssistant, wealth_db
import pandas as pd
from typing import List, Tuple, Optional, Dict
import plotly.graph_objects as go
# Initialize assistant
assistant = WealthManagementAssistant()

# Function to handle chat interactions
def chat(message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """Handle chat interactions."""
    history = history or []

    try:
        response = assistant.generate_response(message)
        history.append((message, response))
        return "", history
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        history.append((message, error_msg))
        return "", history

# Function to update compliance setting
def update_compliance(enabled: bool) -> None:
    """Update compliance checking setting."""
    assistant.compliance_checked = enabled
    return None

# Function to generate client dashboard
def generate_client_dashboard(client_name: str) -> Tuple[Optional[go.Figure], ...]:
    """Generate visualization dashboard for a client."""
    if not client_name:
        return None, None, None, None, None

    return assistant.get_client_visualizations(client_name)

# Gradio Interface
with gr.Blocks(title="Wealth Management AI", theme="soft") as demo:
    gr.Markdown("""
    # 🏦 Enhanced Wealth Management AI Assistant
    *Advanced analytics with interactive visualizations and comprehensive client insights*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            # Database Info
            gr.Markdown("### 📊 Client Database")
            db_preview = gr.Dataframe(
                value=pd.DataFrame([
                    {"Client": "John Smith", "Net Worth": "$4.5M", "Risk": "Moderate", "Accounts": "3"},
                    {"Client": "Emily Davis", "Net Worth": "$2.2M", "Risk": "Conservative", "Accounts": "2"}
                ]),
                interactive=False
            )

            # Document Upload
            gr.Markdown("### 📂 Document Analysis")
            file_input = gr.File(file_count="multiple", file_types=[".pdf"])
            upload_btn = gr.Button("Process Documents", variant="primary")
            doc_status = gr.Textbox(label="Upload Status", interactive=False)
            doc_list = gr.Dataframe(headers=["Document"], interactive=False, visible=False)

            # Compliance Toggle
            gr.Markdown("### ⚖️ Compliance Controls")
            compliance_toggle = gr.Checkbox(
                label="Enable Strict Compliance Checking",
                value=False,
                info="SEC/FINRA regulation checks"
            )

            # Visualization Controls
            gr.Markdown("### 📈 Quick Visualizations")
            client_selector = gr.Dropdown(
                choices=["John Smith", "Emily Davis"],
                label="Select Client",
                value="John Smith"
            )
            viz_btn = gr.Button("Generate Client Dashboard", variant="secondary")

        with gr.Column(scale=2):
            # Chat Interface
            gr.Markdown("### 💬 AI Wealth Advisor")
            chatbot = gr.Chatbot(height=400, show_label=False)
            msg = gr.Textbox(
                label="Query",
                placeholder="Ask about portfolios, risk analysis, or request visualizations...",
                lines=2
            )

            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear Chat", variant="secondary")

            # Example Queries
            gr.Markdown("### 💡 Example Queries")
            gr.Examples(
                examples=[
                    "Show me John Smith's complete portfolio analysis with charts",
                    "Compare Emily Davis' risk metrics to market benchmarks",
                    "Create a performance comparison for both clients",
                    "Generate sector allocation analysis for John Smith",
                    "What are the key risk factors in Emily's portfolio?",
                    "Show market overview dashboard"
                ],
                inputs=msg
            )

    # Visualization Dashboard
    gr.Markdown("## 📊 Interactive Analytics Dashboard")

    with gr.Tab("Client Portfolio Analysis"):
        with gr.Row():
            portfolio_plot = gr.Plot(label="Portfolio Allocation")
            sector_plot = gr.Plot(label="Sector Breakdown")

        with gr.Row():
            performance_plot = gr.Plot(label="Performance vs Benchmark")
            risk_plot = gr.Plot(label="Risk Profile")

        account_plot = gr.Plot(label="Account Breakdown")

    with gr.Tab("Market Overview"):
        market_dashboard = gr.Plot(label="Market Analysis Dashboard")

        # Auto-load market dashboard
        market_dashboard.value = assistant.create_market_comparison_dashboard()

    # Wire up event handlers
    upload_btn.click(
        assistant.process_uploaded_files,
        inputs=file_input,
        outputs=[doc_status, doc_list]
    )

    compliance_toggle.change(
        update_compliance,
        inputs=compliance_toggle,
        outputs=[]
    )

    viz_btn.click(
        generate_client_dashboard,
        inputs=client_selector,
        outputs=[portfolio_plot, sector_plot, performance_plot, risk_plot, account_plot]
    )

    submit_btn.click(
        chat,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )

    msg.submit(
        chat,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )

    clear_btn.click(
        lambda: ([], ""),
        outputs=[chatbot, msg]
    )

# Launch the application
if __name__ == "__main__":
    demo.launch(share=True, debug=True)
