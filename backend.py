from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from PyPDF2 import PdfReader
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta
import io
import base64
import warnings
from dotenv import load_dotenv

# Suppress warnings and configuration
warnings.filterwarnings("ignore")

# Configuration
GOOGLE_API_KEY = "AIzaSyD83yavELzObxwvUrQy8VyZcZ9-TzlFWlU"
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize models
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Enhanced Simulated Wealth Management Database with Time Series Data
def create_simulated_database():
    """Create a comprehensive simulated wealth management database."""
    # Generate historical performance data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')

    clients = {
        "client_001": {
            "name": "John Smith",
            "net_worth": 4500000,
            "risk_tolerance": "Moderate",
            "goals": ["Retirement at 60", "College fund for grandchildren"],
            "portfolio": {
                "stocks": 55,
                "bonds": 30,
                "alternatives": 10,
                "cash": 5
            },
            "accounts": [
                {"id": "acct_1", "type": "Brokerage", "balance": 2800000},
                {"id": "acct_2", "type": "IRA", "balance": 1200000},
                {"id": "acct_3", "type": "Trust", "balance": 500000}
            ],
            "advisors": ["Sarah Johnson", "Michael Chen"],
            "last_review": "2023-11-15",
            "historical_performance": {
                "dates": [d.strftime('%Y-%m-%d') for d in dates],
                "portfolio_values": [4200000 + i*25000 + np.random.normal(0, 50000) for i in range(len(dates))],
                "benchmark_values": [4200000 + i*20000 + np.random.normal(0, 40000) for i in range(len(dates))]
            },
            "sector_allocation": {
                "Technology": 25,
                "Healthcare": 15,
                "Financial Services": 12,
                "Consumer Discretionary": 10,
                "Energy": 8,
                "Utilities": 7,
                "Real Estate": 6,
                "Materials": 5,
                "Other": 12
            }
        },
        "client_002": {
            "name": "Emily Davis",
            "net_worth": 2200000,
            "risk_tolerance": "Conservative",
            "goals": ["Preserve capital", "Charitable giving"],
            "portfolio": {
                "stocks": 30,
                "bonds": 50,
                "alternatives": 15,
                "cash": 5
            },
            "accounts": [
                {"id": "acct_4", "type": "Brokerage", "balance": 1500000},
                {"id": "acct_5", "type": "IRA", "balance": 700000}
            ],
            "advisors": ["Michael Chen"],
            "last_review": "2023-10-20",
            "historical_performance": {
                "dates": [d.strftime('%Y-%m-%d') for d in dates],
                "portfolio_values": [2100000 + i*8000 + np.random.normal(0, 20000) for i in range(len(dates))],
                "benchmark_values": [2100000 + i*7000 + np.random.normal(0, 18000) for i in range(len(dates))]
            },
            "sector_allocation": {
                "Government Bonds": 30,
                "Corporate Bonds": 20,
                "Utilities": 12,
                "Consumer Staples": 10,
                "Healthcare": 8,
                "Dividend Stocks": 8,
                "REITs": 7,
                "Cash Equivalents": 5
            }
        }
    }

    market_data = {
        "SP500": {"ytd_return": 12.4, "pe_ratio": 22.3, "volatility": 16.2},
        "Bonds": {"ytd_return": 3.2, "duration": 6.5, "yield": 4.1},
        "RealEstate": {"ytd_return": 5.7, "cap_rate": 4.2, "occupancy": 92.5},
        "Commodities": {"ytd_return": -2.1, "volatility": 24.8}
    }

    risk_metrics = {
        "client_001": {
            "sharpe_ratio": 1.45,
            "max_drawdown": -8.2,
            "beta": 1.15,
            "alpha": 2.3,
            "var_95": -125000,
            "volatility": 15.7
        },
        "client_002": {
            "sharpe_ratio": 0.98,
            "max_drawdown": -4.1,
            "beta": 0.65,
            "alpha": 1.1,
            "var_95": -45000,
            "volatility": 9.2
        }
    }

    return {"clients": clients, "market_data": market_data, "risk_metrics": risk_metrics}

wealth_db = create_simulated_database()

class WealthVisualizationMixin:
    """Mixin class for creating financial visualizations."""

    def create_portfolio_pie_chart(self, client_id: str) -> Optional[go.Figure]:
        """Create portfolio allocation pie chart."""
        if client_id not in wealth_db["clients"]:
            return None
            
        client = wealth_db["clients"][client_id]
        portfolio = client["portfolio"]

        fig = px.pie(
            values=list(portfolio.values()),
            names=list(portfolio.keys()),
            title=f"{client['name']} - Portfolio Allocation",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=True, height=400)
        return fig

    def create_sector_allocation_chart(self, client_id: str) -> Optional[go.Figure]:
        """Create sector allocation bar chart."""
        if client_id not in wealth_db["clients"]:
            return None
            
        client = wealth_db["clients"][client_id]
        if "sector_allocation" not in client:
            return None

        sectors = list(client["sector_allocation"].keys())
        allocations = list(client["sector_allocation"].values())

        fig = px.bar(
            x=allocations,
            y=sectors,
            orientation='h',
            title=f"{client['name']} - Sector Allocation",
            labels={'x': 'Allocation (%)', 'y': 'Sectors'},
            color=allocations,
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=500, showlegend=False)
        return fig

    def create_performance_comparison(self, client_id: str) -> Optional[go.Figure]:
        """Create performance comparison line chart."""
        if client_id not in wealth_db["clients"]:
            return None
            
        client = wealth_db["clients"][client_id]
        if "historical_performance" not in client:
            return None

        hist_data = client["historical_performance"]
        dates = pd.to_datetime(hist_data["dates"])

        fig = go.Figure()

        # Portfolio performance
        fig.add_trace(go.Scatter(
            x=dates,
            y=hist_data["portfolio_values"],
            mode='lines',
            name='Portfolio',
            line=dict(color='blue', width=3)
        ))

        # Benchmark performance
        fig.add_trace(go.Scatter(
            x=dates,
            y=hist_data["benchmark_values"],
            mode='lines',
            name='Benchmark',
            line=dict(color='red', width=2, dash='dash')
        ))

        fig.update_layout(
            title=f"{client['name']} - Portfolio vs Benchmark Performance",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified',
            height=400
        )

        return fig

    def create_risk_metrics_radar(self, client_id: str) -> Optional[go.Figure]:
        """Create risk metrics radar chart."""
        if client_id not in wealth_db["risk_metrics"]:
            return None

        metrics = wealth_db["risk_metrics"][client_id]
        client_name = wealth_db["clients"][client_id]["name"]

        # Normalize metrics for radar chart (0-100 scale)
        categories = ['Sharpe Ratio', 'Alpha', 'Beta', 'Max Drawdown', 'Volatility']
        values = [
            min(metrics["sharpe_ratio"] * 30, 100),  # Scale Sharpe ratio
            min(metrics["alpha"] * 20, 100),         # Scale Alpha
            100 - abs(metrics["beta"] - 1) * 50,     # Beta closer to 1 is better
            100 + metrics["max_drawdown"],           # Max drawdown (negative)
            max(100 - abs(metrics.get("volatility", 15)), 0)  # Lower volatility is better
        ]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=client_name,
            line=dict(color='blue')
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title=f"{client_name} - Risk Profile Analysis",
            height=400
        )

        return fig
        
    def create_account_breakdown_chart(self, client_id: str) -> Optional[go.Figure]:
        """Create account breakdown chart."""
        if client_id not in wealth_db["clients"]:
            return None
            
        client = wealth_db["clients"][client_id]
        accounts = client["accounts"]

        account_types = [acc["type"] for acc in accounts]
        balances = [acc["balance"] for acc in accounts]

        fig = px.bar(
            x=account_types,
            y=balances,
            title=f"{client['name']} - Account Breakdown",
            labels={'x': 'Account Type', 'y': 'Balance ($)'},
            color=balances,
            color_continuous_scale='blues'
        )

        fig.update_layout(
            yaxis_tickformat='$,.0f',
            height=400,
            showlegend=False
        )

        return fig

    def create_market_comparison_dashboard(self) -> go.Figure:
        """Create market comparison dashboard."""
        market_data = wealth_db["market_data"]

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('YTD Returns', 'Risk Metrics', 'Valuations', 'Yields'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )

        # YTD Returns
        assets = list(market_data.keys())
        returns = [market_data[asset]["ytd_return"] for asset in assets]

        fig.add_trace(
            go.Bar(x=assets, y=returns, name="YTD Return (%)", marker_color='lightblue'),
            row=1, col=1
        )

        # Risk Metrics (Volatility where available)
        volatilities = [market_data[asset].get("volatility", 0) for asset in assets]
        fig.add_trace(
            go.Bar(x=assets, y=volatilities, name="Volatility (%)", marker_color='lightcoral'),
            row=1, col=2
        )

        # Risk-Return Scatter
        fig.add_trace(
            go.Scatter(
                x=volatilities,
                y=returns,
                mode='markers+text',
                text=assets,
                textposition="top center",
                name="Risk-Return",
                marker=dict(size=12, color='green')
            ),
            row=2, col=1
        )

        # Yields/Rates
        yields = [
            market_data["SP500"]["pe_ratio"],
            market_data["Bonds"]["yield"],
            market_data["RealEstate"]["cap_rate"],
            0  # Commodities don't have yield
        ]

        fig.add_trace(
            go.Bar(x=assets, y=yields, name="Yields/Ratios", marker_color='gold'),
            row=2, col=2
        )

        fig.update_layout(height=600, showlegend=False, title_text="Market Overview Dashboard")
        return fig

class WealthManagementAssistant(WealthVisualizationMixin):
    """Main wealth management assistant class."""
    
    def __init__(self):
        """Initialize the assistant with default settings."""
        self.vector_db = None
        self.uploaded_files = []
        self.compliance_checked = False
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.scope_description = """
        This assistant specializes in wealth management topics including:
        - Client portfolio analysis with visualizations
        - Risk assessment and metrics
        - Investment strategy and allocation
        - Market comparisons and benchmarking
        - Account-specific queries
        - Financial planning with charts
        - Performance analysis
        - Tax strategies
        - Estate planning
        """

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""

    def process_uploaded_files(self, files: List[str]) -> Tuple[str, List[Tuple[str, str]]]:
        """Process uploaded PDF files into a vector database."""
        if not files:
            return "No files uploaded", []

        self.uploaded_files = files
        documents = []
        metadatas = []

        for i, file_path in enumerate(files):
            try:
                text = self.extract_text_from_pdf(file_path)
                if not text:
                    continue
                    
                chunks = self.text_splitter.split_text(text)
                doc_type = f"UploadedDoc_{i+1}"
                metadata = [{"source": file_path, "type": doc_type} for _ in chunks]

                documents.extend(chunks)
                metadatas.extend(metadata)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

        if documents:
            try:
                self.vector_db = Chroma.from_texts(
                    documents,
                    embeddings,
                    metadatas=metadatas
                )
                return "Documents processed successfully!", [
                    (f"Document {i+1}: {os.path.basename(file)}", "")
                    for i, file in enumerate(files)
                ]
            except Exception as e:
                return f"Error creating vector database: {str(e)}", []
                
        return "No documents processed", []

    def check_compliance(self, message: str) -> bool:
        """Check if a message complies with regulations."""
        if not self.compliance_checked:
            return True

        prompt = f"""Analyze this wealth management query for compliance issues (SEC/FINRA):

        Query: {message}

        Identify regulatory concerns or privacy risks. Respond with ONLY "APPROVED" or "REJECTED: [reason]":"""

        try:
            response = llm.invoke(prompt).content
            return response.startswith("APPROVED")
        except Exception as e:
            print(f"Compliance check error: {e}")
            return True  # Default to approved if check fails

    def is_in_scope(self, query: str) -> bool:
        """Check if the query falls within wealth management scope."""
        prompt = f"""Determine if this query falls within wealth management scope:

        Scope Description: {self.scope_description}

        Query: {query}

        Respond with ONLY "YES" or "NO":"""

        try:
            response = llm.invoke(prompt).content.strip().upper()
            return response == "YES"
        except Exception as e:
            print(f"Scope check error: {e}")
            return True  # Default to in-scope if check fails

    def query_wealth_db(self, query: str) -> str:
        """Query the simulated wealth database."""
        db_context = f"""
        WEALTH DATABASE CONTEXT:
        {json.dumps(wealth_db, indent=2)}
        """

        prompt = f"""You are a wealth management AI with access to this database:
        {db_context}

        Analyze the following query and provide a detailed response with specific numbers:

        Query: {query}

        Response Guidelines:
        1. Always reference specific client data when available
        2. Include relevant market data comparisons
        3. Highlight any anomalies or opportunities
        4. Format numbers with proper formatting ($ and commas)
        5. Suggest next steps when appropriate
        6. Mention when visualizations would be helpful"""

        try:
            return llm.invoke(prompt).content
        except Exception as e:
            return f"Error querying database: {str(e)}"

    def query_documents(self, query: str) -> str:
        """Query uploaded documents if available."""
        if not self.vector_db:
            return ""

        try:
            docs = self.vector_db.similarity_search(query, k=3)
            if not docs:
                return ""

            context = "\n\n--- DOCUMENT EXCERPTS ---\n\n" + \
                    "\n\n".join(f"DOCUMENT {i+1}:\n{doc.page_content}"
                              for i, doc in enumerate(docs))

            prompt = f"""Analyze these document excerpts in relation to the query:

            {context}

            Query: {query}

            Provide specific insights from documents, or state if irrelevant."""

            return llm.invoke(prompt).content
        except Exception as e:
            print(f"Error querying documents: {e}")
            return ""

    def generate_response(self, query: str) -> str:
        """Generate a comprehensive response combining DB and documents."""
        # First check if query is in scope
        if not self.is_in_scope(query):
            return "This prompt is out of scope for wealth management assistance."

        # Check compliance first
        if not self.check_compliance(query):
            return "COMPLIANCE REJECTION: This query cannot be processed due to regulatory concerns."

        # Query wealth database
        db_response = self.query_wealth_db(query)

        # Query documents if available
        doc_response = self.query_documents(query) if self.vector_db else ""

        # Combine responses
        if doc_response:
            prompt = f"""Combine these responses into one coherent answer:

            DATABASE RESPONSE:
            {db_response}

            DOCUMENT ANALYSIS:
            {doc_response}

            Create a professional wealth management response that:
            1. Begins with the most relevant information
            2. Integrates both sources seamlessly
            3. Ends with recommended next steps
            4. Mentions available visualizations when relevant"""

            try:
                return llm.invoke(prompt).content
            except Exception as e:
                print(f"Response combination error: {e}")
                return db_response

        return db_response

    def get_client_visualizations(self, client_name: str) -> Tuple[Optional[go.Figure], ...]:
        """Get all visualizations for a specific client."""
        # Find client by name
        client_id = None
        for cid, client in wealth_db["clients"].items():
            if client["name"].lower() == client_name.lower():
                client_id = cid
                break

        if not client_id:
            return None, None, None, None, None

        # Generate all charts
        portfolio_chart = self.create_portfolio_pie_chart(client_id)
        sector_chart = self.create_sector_allocation_chart(client_id)
        performance_chart = self.create_performance_comparison(client_id)
        risk_chart = self.create_risk_metrics_radar(client_id)
        account_chart = self.create_account_breakdown_chart(client_id)

        return portfolio_chart, sector_chart, performance_chart, risk_chart, account_chart
