import streamlit as st

def load_css():
    """
    Injects custom CSS to style the Streamlit app with a premium, high-tech dark appearance
    featuring glassmorphic cards, gradient headings, and modern typography.
    """
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');
    
    /* Global Typography */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6, [data-testid="stHeader"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Gradient Header Text */
    .gradient-header {
        background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    
    .gradient-sub {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Glassmorphic card styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }

    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(168, 85, 247, 0.15);
        border-color: rgba(168, 85, 247, 0.4);
    }
    
    .glass-card h3 {
        color: #f1f5f9;
        margin-top: 0;
        margin-bottom: 12px;
        font-size: 1.3rem;
        font-weight: 600;
    }

    /* Feature Badge / Tag list */
    .badge {
        display: inline-block;
        background: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 4px 4px 4px 0;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    
    .badge-blue {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    .badge-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .badge-orange {
        background: rgba(249, 115, 22, 0.15);
        color: #fb923c;
        border: 1px solid rgba(249, 115, 22, 0.3);
    }

    /* Formula display block */
    .formula-box {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        margin: 15px 0;
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
        color: #e2e8f0;
        text-align: center;
        box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.2);
    }
    
    .formula-highlight {
        color: #a855f7;
        font-weight: bold;
    }

    /* Profile Detail Items */
    .profile-item {
        margin-bottom: 8px;
        font-size: 0.95rem;
    }
    
    .profile-label {
        font-weight: 600;
        color: #94a3b8;
        display: inline-block;
        width: 140px;
    }
    
    .profile-value {
        color: #f1f5f9;
    }
    
    /* Metrics Custom Override */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.02);
        padding: 15px 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }
    
    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
