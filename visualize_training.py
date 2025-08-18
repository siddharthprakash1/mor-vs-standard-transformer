import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from scipy import stats
from scipy.stats import pearsonr

# Enhanced colors for 4 models with dark theme compatibility
MOR_COLOR = '#e74c3c'           # red
STD_COLOR = '#3498db'           # blue  
GEMMA_MOR_COLOR = '#f39c12'     # orange
GEMMA_STD_COLOR = '#2ecc71'     # green
POS_COLOR = 'rgba(39, 174, 96, 0.25)'    # greenish (better)
NEG_COLOR = 'rgba(231, 76, 60, 0.25)'    # reddish (worse)

# Model color mapping
MODEL_COLORS = {
    'MoR_400M': MOR_COLOR,
    'Standard': STD_COLOR,
    'Gemma_MoR': GEMMA_MOR_COLOR,
    'Gemma_Standard': GEMMA_STD_COLOR
}

# Dark theme colors
DARK_THEME = {
    'bg_color': '#0E1117',
    'paper_bg': '#1E2329',
    'text_color': '#FAFAFA',
    'grid_color': '#2F3349',
    'accent_color': '#FF6B6B',
    'secondary_color': '#4ECDC4'
}

def apply_dark_theme_to_fig(fig):
    """Apply dark theme to plotly figures"""
    fig.update_layout(
        plot_bgcolor=DARK_THEME['bg_color'],
        paper_bgcolor=DARK_THEME['paper_bg'],
        font_color=DARK_THEME['text_color'],
        font_family="Arial, sans-serif",
        title_font_color=DARK_THEME['text_color'],
        legend=dict(
            bgcolor=DARK_THEME['paper_bg'],
            bordercolor=DARK_THEME['grid_color'],
            font_color=DARK_THEME['text_color']
        )
    )
    
    # Update axes
    fig.update_xaxes(
        gridcolor=DARK_THEME['grid_color'],
        color=DARK_THEME['text_color'],
        tickfont_color=DARK_THEME['text_color']
    )
    fig.update_yaxes(
        gridcolor=DARK_THEME['grid_color'],
        color=DARK_THEME['text_color'],
        tickfont_color=DARK_THEME['text_color']
    )
    
    return fig

# Set page config
st.set_page_config(
    page_title="Complete 4-Model Training Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme and better styling
st.markdown(
    """
<style>
    .stApp {
        background-color: #0E1117;
    }
    
    .main {
        background-color: #0E1117;
    }
    
    .css-1d391kg {
        background-color: #1E2329;
    }
    
    .stSelectbox > div > div {
        background-color: #1E2329;
        color: #FAFAFA;
    }
    
    .stMultiSelect > div > div {
        background-color: #1E2329;
        color: #FAFAFA;
    }
    
    .stSlider > div > div > div {
        color: #FAFAFA;
    }
    
    .main-header {
        font-size: 3rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #1E2329 0%, #2F3349 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #FAFAFA;
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid #4ECDC4;
    }
    .section-header {
        color: #4ECDC4;
        border-bottom: 3px solid #4ECDC4;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    .highlight-box {
        background: #1E2329;
        border-left: 5px solid #4ECDC4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        color: #FAFAFA;
    }
    .dark-insight-box {
        background: #1E2329;
        color: #FAFAFA;
        border-left: 5px solid #4ECDC4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1E2329;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #FAFAFA;
        background-color: #1E2329;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4ECDC4;
        color: #0E1117;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #FAFAFA !important;
    }
    
    .stMarkdown {
        color: #FAFAFA;
    }
    
    .stAlert {
        background-color: #1E2329;
        color: #FAFAFA;
    }
    
    div[data-testid="stSidebar"] {
        background-color: #1E2329;
    }
    
    div[data-testid="stSidebar"] .stMarkdown {
        color: #FAFAFA;
    }
    
    .stDataFrame {
        background-color: #1E2329;
    }
    
    .stJson {
        background-color: #1E2329;
        color: #FAFAFA;
    }
    
    .stMetric {
        background-color: #1E2329;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #2F3349;
    }
    
    .stMetric > div {
        color: #FAFAFA;
    }
    
    .stDownloadButton > button {
        background-color: #4ECDC4;
        color: #0E1117;
        border: none;
    }
    
    .stDownloadButton > button:hover {
        background-color: #45B7D1;
    }
    
    .model-card {
        background: linear-gradient(135deg, #1E2329 0%, #2F3349 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: #FAFAFA;
        margin: 1rem 0;
        border: 2px solid;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .model-card-mor { border-color: #e74c3c; }
    .model-card-std { border-color: #3498db; }
    .model-card-gemma-mor { border-color: #f39c12; }
    .model-card-gemma-std { border-color: #2ecc71; }
</style>
""",
    unsafe_allow_html=True,
)

# Load data function for ALL 4 models
@st.cache_data
def load_all_training_data():
    """Load and process training data from ALL JSON files"""
    models_data = {}
    
    try:
        # 1. MoR 400M data
        with open('training_metrics_400m.json', 'r') as f:
            mor_400m_data = json.load(f)
        models_data['MoR_400M'] = pd.DataFrame(mor_400m_data)
        models_data['MoR_400M']['model'] = 'MoR_400M'

        # 2. Standard data  
        with open('training_metrics.json', 'r') as f:
            standard_data = json.load(f)
        models_data['Standard'] = pd.DataFrame(standard_data)
        models_data['Standard']['model'] = 'Standard'

        # 3. Gemma MoR data
        with open('gemma_mor_training_stats_20250817_162231.json', 'r') as f:
            gemma_mor_raw = json.load(f)
        
        # Process Gemma MoR training progress
        gemma_mor_progress = []
        for entry in gemma_mor_raw['training_progress']:
            gemma_mor_progress.append({
                'steps': entry['step'],
                'total_loss': entry['loss'],
                'avg_loss': entry['avg_loss'], 
                'lr': entry['learning_rate'],
                'gpu_memory_gb': entry['gpu_memory_used_gb'],
                'timestamp': entry['timestamp']
            })
        models_data['Gemma_MoR'] = pd.DataFrame(gemma_mor_progress)
        models_data['Gemma_MoR']['model'] = 'Gemma_MoR'

        # 4. Gemma Standard data
        with open('standard_gemma_training_stats_20250817_232143.json', 'r') as f:
            gemma_std_raw = json.load(f)
        
        # Process Gemma Standard training progress
        gemma_std_progress = []
        for entry in gemma_std_raw['training_progress']:
            gemma_std_progress.append({
                'steps': entry['step'],
                'total_loss': entry['loss'],
                'avg_loss': entry['avg_loss'],
                'lr': entry['learning_rate'],
                'gpu_memory_gb': entry['gpu_memory_used_gb'],
                'perplexity': entry['perplexity'],
                'timestamp': entry['timestamp']
            })
        models_data['Gemma_Standard'] = pd.DataFrame(gemma_std_progress)
        models_data['Gemma_Standard']['model'] = 'Gemma_Standard'

        # 5. Training loss data (epoch-based)
        with open('standard_training_data.json', 'r') as f:
            standard_train_data = json.load(f)
        with open('mor_training_data.json', 'r') as f:
            mor_train_data = json.load(f)
        with open('gemma_mor_training_data.json', 'r') as f:
            gemma_mor_train_data = json.load(f)

        # Create training DataFrames with fixed range conversion
        training_data = {}
        training_data['Standard'] = pd.DataFrame({
            'epoch': list(range(len(standard_train_data['train_losses']))),
            'train_loss': standard_train_data['train_losses'],
            'model': 'Standard',
        })
        
        training_data['MoR'] = pd.DataFrame({
            'epoch': list(range(len(mor_train_data['train_losses']))),
            'train_loss': mor_train_data['train_losses'],
            'model': 'MoR',
        })
        
        training_data['Gemma_MoR'] = pd.DataFrame({
            'epoch': list(range(len(gemma_mor_train_data['train_losses']))),
            'train_loss': gemma_mor_train_data['train_losses'],
            'model': 'Gemma_MoR',
        })

        return models_data, training_data

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Enhanced model specifications for all 4 models
@st.cache_data
def get_all_model_specs():
    """Get specifications for all 4 models"""
    return {
        'MoR_400M': {
            'name': 'MoR 400M',
            'parameters': 365_615_105,
            'size_mb': 1394.7,
            'd_model': 1280,
            'n_heads': 20,
            'd_ff': 5120,
            'n_layers_shared': 12,
            'max_recursions': 3,
            'effective_depth': 36,
            'architecture_type': 'Mixture of Recursions',
            'vocab_size': 50257,
            'color': MOR_COLOR,
        },
        'Standard': {
            'name': 'Standard Transformer',
            'parameters': 405_613_568,
            'size_mb': 1547.3,
            'd_model': 1024,
            'n_heads': 16,
            'd_ff': 4096,
            'n_layers': 24,
            'effective_depth': 24,
            'architecture_type': 'Standard Transformer',
            'vocab_size': 50257,
            'color': STD_COLOR,
        },
        'Gemma_MoR': {
            'name': 'Gemma 3 270M with MoR',
            'parameters': 130_121_860,  # trainable parameters
            'total_parameters': 2_744_463_748,
            'size_mb': 10485,  # estimated from total params
            'd_model': 2304,
            'num_shared_layers': 2,
            'num_routing_tokens': 4,
            'routing_temperature': 1.0,
            'architecture_type': 'Gemma + MoR',
            'color': GEMMA_MOR_COLOR,
        },
        'Gemma_Standard': {
            'name': 'Standard Gemma 3 270M',
            'parameters': 268_098_176,
            'size_mb': 1024,  # estimated
            'architecture_type': 'Standard Gemma',
            'color': GEMMA_STD_COLOR,
        },
    }

def create_comprehensive_loss_comparison(models_data):
    """Create comprehensive loss comparison for all 4 models"""
    fig = go.Figure()
    
    for model_name, df in models_data.items():
        if 'total_loss' in df.columns and 'steps' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['steps'],
                    y=df['total_loss'],
                    name=model_name.replace('_', ' '),
                    line=dict(color=MODEL_COLORS[model_name], width=3),
                    mode='lines+markers',
                    marker=dict(size=4)
                )
            )
    
    fig.update_layout(
        title="Complete Training Loss Comparison - All 4 Models",
        title_x=0.5,
        xaxis_title="Training Steps",
        yaxis_title="Total Loss",
        height=600,
        hovermode='x unified'
    )
    
    return apply_dark_theme_to_fig(fig)

def create_model_architecture_radar(specs):
    """Create radar chart comparing all model architectures"""
    
    # Normalize parameters to millions for comparison
    categories = ['Parameters (M)', 'Model Size (GB)', 'd_model/100', 'Layers/Depth']
    
    fig = go.Figure()
    
    for model_name, spec in specs.items():
        if model_name in ['MoR_400M', 'Standard']:  # Full spec models
            values = [
                spec['parameters'] / 1e6,
                spec['size_mb'] / 1000,
                spec['d_model'] / 100,
                spec.get('effective_depth', spec.get('n_layers', 24))
            ]
        elif model_name == 'Gemma_MoR':
            values = [
                spec['parameters'] / 1e6,
                spec['size_mb'] / 1000,
                spec['d_model'] / 100,
                spec['num_shared_layers'] * 4  # approximate effective depth
            ]
        else:  # Gemma_Standard
            values = [
                spec['parameters'] / 1e6,
                spec['size_mb'] / 1000,
                270,  # approximate d_model
                24    # approximate layers
            ]
            
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=spec['name'],
                line_color=spec['color'],
                opacity=0.7
            )
        )
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 500],  # Adjust range as needed
                color=DARK_THEME['text_color'],
                gridcolor=DARK_THEME['grid_color']
            ),
            angularaxis=dict(
                color=DARK_THEME['text_color'],
                gridcolor=DARK_THEME['grid_color']
            ),
            bgcolor=DARK_THEME['bg_color']
        ),
        title="Model Architecture Comparison - All 4 Models",
        title_x=0.5,
        height=600
    )
    
    return apply_dark_theme_to_fig(fig)

def create_performance_leaderboard(models_data, specs):
    """Create performance leaderboard with key metrics"""
    leaderboard_data = []
    
    for model_name, df in models_data.items():
        if 'total_loss' in df.columns and len(df) > 0:
            final_loss = float(df['total_loss'].iloc[-1])
            min_loss = float(df['total_loss'].min())
            mean_loss = float(df['total_loss'].mean())
            std_loss = float(df['total_loss'].std())
            improvement = float(df['total_loss'].iloc[0] - df['total_loss'].iloc[-1]) if len(df) > 1 else 0
            
            # Parameter efficiency
            params_m = specs[model_name]['parameters'] / 1e6
            param_efficiency = final_loss / params_m
            
            leaderboard_data.append({
                'Model': specs[model_name]['name'],
                'Final Loss': final_loss,
                'Min Loss': min_loss,
                'Mean Loss': mean_loss,
                'Loss Std': std_loss,
                'Improvement': improvement,
                'Parameters (M)': params_m,
                'Param Efficiency': param_efficiency,
                'Architecture': specs[model_name]['architecture_type']
            })
    
    leaderboard_df = pd.DataFrame(leaderboard_data)
    leaderboard_df = leaderboard_df.sort_values('Final Loss')  # Best (lowest) loss first
    
    return leaderboard_df

def create_training_efficiency_comparison(models_data):
    """Create training efficiency comparison"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Loss Convergence', 'Learning Rate Schedules', 
                       'Memory Usage', 'Loss Distributions'),
    )
    
    # Loss convergence (smoothed)
    for model_name, df in models_data.items():
        if 'total_loss' in df.columns and 'steps' in df.columns and len(df) > 5:
            smoothed = df['total_loss'].rolling(window=min(5, len(df)//2), min_periods=1).mean()
            fig.add_trace(
                go.Scatter(
                    x=df['steps'],
                    y=smoothed,
                    name=model_name.replace('_', ' '),
                    line=dict(color=MODEL_COLORS[model_name], width=2)
                ),
                row=1, col=1
            )
    
    # Learning rate schedules
    for model_name, df in models_data.items():
        if 'lr' in df.columns and 'steps' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['steps'],
                    y=df['lr'],
                    name=f"{model_name} LR",
                    line=dict(color=MODEL_COLORS[model_name], width=2, dash='dot'),
                    showlegend=False
                ),
                row=1, col=2
            )
    
    # Memory usage
    for model_name, df in models_data.items():
        if 'gpu_memory_gb' in df.columns and 'steps' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['steps'],
                    y=df['gpu_memory_gb'],
                    name=f"{model_name} Memory",
                    line=dict(color=MODEL_COLORS[model_name], width=2, dash='dash'),
                    showlegend=False
                ),
                row=2, col=1
            )
    
    # Loss distributions (box plots)
    for model_name, df in models_data.items():
        if 'total_loss' in df.columns:
            fig.add_trace(
                go.Box(
                    y=df['total_loss'],
                    name=model_name.replace('_', ' '),
                    marker_color=MODEL_COLORS[model_name],
                    showlegend=False
                ),
                row=2, col=2
            )
    
    fig.update_layout(
        height=800,
        title_text="Training Efficiency Comparison - All Models",
        title_x=0.5
    )
    
    return apply_dark_theme_to_fig(fig)

def create_pairwise_comparison(models_data, model_a, model_b):
    """Create detailed pairwise comparison between two selected models"""
    if model_a not in models_data or model_b not in models_data:
        return None
        
    df_a = models_data[model_a]
    df_b = models_data[model_b]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(f'{model_a} vs {model_b} Loss', 'Loss Difference',
                       'Learning Rate Comparison', 'Performance Scatter'),
    )
    
    # Direct comparison
    if 'total_loss' in df_a.columns and 'total_loss' in df_b.columns:
        fig.add_trace(
            go.Scatter(
                x=df_a['steps'] if 'steps' in df_a.columns else list(range(len(df_a))),
                y=df_a['total_loss'],
                name=model_a,
                line=dict(color=MODEL_COLORS[model_a], width=3)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_b['steps'] if 'steps' in df_b.columns else list(range(len(df_b))),
                y=df_b['total_loss'],
                name=model_b,
                line=dict(color=MODEL_COLORS[model_b], width=3)
            ),
            row=1, col=1
        )
    
    # Loss difference (if same length)
    if len(df_a) == len(df_b) and 'total_loss' in df_a.columns and 'total_loss' in df_b.columns:
        diff = df_a['total_loss'].values - df_b['total_loss'].values
        steps_a = df_a['steps'] if 'steps' in df_a.columns else list(range(len(df_a)))
        
        fig.add_trace(
            go.Scatter(
                x=steps_a,
                y=diff,
                name=f'{model_a} - {model_b}',
                line=dict(color='white', width=2),
                fill='tozeroy'
            ),
            row=1, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
    
    # Learning rates
    for model_name, df, row_col in [(model_a, df_a, (2, 1)), (model_b, df_b, (2, 1))]:
        if 'lr' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['steps'] if 'steps' in df.columns else list(range(len(df))),
                    y=df['lr'],
                    name=f'{model_name} LR',
                    line=dict(color=MODEL_COLORS[model_name], width=2),
                    showlegend=False
                ),
                row=row_col[0], col=row_col[1]
            )
    
    # Performance scatter
    if 'total_loss' in df_a.columns and 'total_loss' in df_b.columns:
        min_len = min(len(df_a), len(df_b))
        fig.add_trace(
            go.Scatter(
                x=df_a['total_loss'][:min_len],
                y=df_b['total_loss'][:min_len],
                mode='markers',
                marker=dict(
                    size=6,
                    color=list(range(min_len)),
                    colorscale='Viridis',
                    showscale=True
                ),
                name='Loss Correlation',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Add diagonal reference line
        max_val = max(df_a['total_loss'][:min_len].max(), df_b['total_loss'][:min_len].max())
        min_val = min(df_a['total_loss'][:min_len].min(), df_b['total_loss'][:min_len].min())
        fig.add_shape(
            type="line",
            x0=min_val, y0=min_val,
            x1=max_val, y1=max_val,
            line=dict(color="red", dash="dash"),
            row=2, col=2
        )
    
    fig.update_layout(
        height=700,
        title_text=f"Detailed Comparison: {model_a} vs {model_b}",
        title_x=0.5
    )
    
    return apply_dark_theme_to_fig(fig)

def create_advanced_statistical_analysis(models_data):
    """Advanced statistical analysis across all models"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Model Performance Violin Plots', 'Correlation Heatmap',
                       'Loss Convergence Rates', 'Statistical Summary'),
    )
    
    # Violin plots
    all_losses = []
    all_models = []
    for model_name, df in models_data.items():
        if 'total_loss' in df.columns:
            all_losses.extend(df['total_loss'].tolist())
            all_models.extend([model_name.replace('_', ' ')] * len(df))
    
    if all_losses:
        df_violin = pd.DataFrame({'loss': all_losses, 'model': all_models})
        for model in df_violin['model'].unique():
            model_data = df_violin[df_violin['model'] == model]
            model_key = model.replace(' ', '_')
            if model_key in MODEL_COLORS:
                fig.add_trace(
                    go.Violin(
                        y=model_data['loss'],
                        name=model,
                        line_color=MODEL_COLORS[model_key],
                        showlegend=False
                    ),
                    row=1, col=1
                )
    
    # Convergence rates (improvement over time)
    for model_name, df in models_data.items():
        if 'total_loss' in df.columns and len(df) > 10:
            # Calculate rolling improvement
            window = min(10, len(df) // 3)
            rolling_improvement = df['total_loss'].rolling(window).apply(
                lambda x: x.iloc[0] - x.iloc[-1] if len(x) == window else 0
            )
            
            steps = list(range(len(rolling_improvement)))  # Fixed range conversion
            
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=rolling_improvement.tolist(),
                    name=model_name.replace('_', ' '),
                    line=dict(color=MODEL_COLORS[model_name], width=2),
                    showlegend=False
                ),
                row=2, col=1
            )
    
    # Statistical summary bar chart
    summary_data = {
        'Model': [],
        'Final Loss': [],
        'Min Loss': [],
        'Mean Loss': [],
        'Std Loss': []
    }
    
    for model_name, df in models_data.items():
        if 'total_loss' in df.columns and len(df) > 0:
            summary_data['Model'].append(model_name.replace('_', ' '))
            summary_data['Final Loss'].append(df['total_loss'].iloc[-1])
            summary_data['Min Loss'].append(df['total_loss'].min())
            summary_data['Mean Loss'].append(df['total_loss'].mean())
            summary_data['Std Loss'].append(df['total_loss'].std())
    
    if summary_data['Model']:
        for i, model in enumerate(summary_data['Model']):
            model_key = model.replace(' ', '_')
            fig.add_trace(
                go.Bar(
                    x=['Final', 'Min', 'Mean', 'Std'],
                    y=[summary_data['Final Loss'][i], summary_data['Min Loss'][i], 
                       summary_data['Mean Loss'][i], summary_data['Std Loss'][i]],
                    name=model,
                    marker_color=MODEL_COLORS.get(model_key, '#888888'),
                    showlegend=False,
                    offsetgroup=i
                ),
                row=2, col=2
            )
    
    fig.update_layout(
        height=800,
        title_text="Advanced Statistical Analysis - All Models",
        title_x=0.5
    )
    
    return apply_dark_theme_to_fig(fig)

def main():
    # Header
    st.markdown(
        '<h1 class="main-header">🧠 Complete 4-Model Training Analysis Dashboard</h1>',
        unsafe_allow_html=True,
    )
    
    st.markdown(
        '<p style="text-align: center; color: #4ECDC4; font-size: 1.2rem;">MoR 400M • Standard Transformer • Gemma MoR • Gemma Standard</p>',
        unsafe_allow_html=True,
    )

    # Load all data
    models_data, training_data = load_all_training_data()
    specs = get_all_model_specs()

    if models_data is None:
        st.error("Failed to load data. Please ensure all JSON files are available.")
        return

    # Sidebar controls
    st.sidebar.title("🎛️ Dashboard Controls")
    st.sidebar.markdown("---")

    # Model selection for comparisons
    available_models = list(models_data.keys())
    selected_models = st.sidebar.multiselect(
        "Select Models to Display",
        available_models,
        default=available_models,
        help="Choose which models to include in comparisons"
    )
    
    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        [
            "📊 Overview Dashboard",
            "🏆 Performance Leaderboard", 
            "📈 Loss Comparison",
            "🏗️ Architecture Analysis",
            "⚡ Training Efficiency",
            "🔍 Pairwise Comparison",
            "📊 Statistical Analysis",
            "💡 Insights & Recommendations"
        ]
    )

    # Filter models based on selection
    filtered_models = {k: v for k, v in models_data.items() if k in selected_models}

    # Analysis sections
    if analysis_type == "📊 Overview Dashboard":
        st.markdown('<h2 class="section-header">📊 Model Overview Dashboard</h2>', unsafe_allow_html=True)
        
        # Model cards
        cols = st.columns(len(selected_models))
        for i, model_name in enumerate(selected_models):
            with cols[i]:
                df = models_data[model_name]
                spec = specs[model_name]
                
                card_class = f"model-card model-card-{model_name.lower().replace('_', '-')}"
                final_loss = float(df['total_loss'].iloc[-1]) if 'total_loss' in df.columns and len(df) > 0 else 0
                
                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <h3 style="color: {spec['color']};">{spec['name']}</h3>
                        <p><strong>Parameters:</strong> {spec['parameters']:,}</p>
                        <p><strong>Final Loss:</strong> {final_loss:.3f}</p>
                        <p><strong>Architecture:</strong> {spec['architecture_type']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Main comparison chart
        if filtered_models:
            st.plotly_chart(create_comprehensive_loss_comparison(filtered_models), use_container_width=True)

    elif analysis_type == "🏆 Performance Leaderboard":
        st.markdown('<h2 class="section-header">🏆 Performance Leaderboard</h2>', unsafe_allow_html=True)
        
        leaderboard_df = create_performance_leaderboard(filtered_models, specs)
        
        if not leaderboard_df.empty:
            st.dataframe(
                leaderboard_df.style.format({
                    'Final Loss': '{:.4f}',
                    'Min Loss': '{:.4f}',
                    'Mean Loss': '{:.4f}',
                    'Loss Std': '{:.4f}',
                    'Improvement': '{:.4f}',
                    'Parameters (M)': '{:.1f}',
                    'Param Efficiency': '{:.6f}',
                }),
                use_container_width=True
            )
            
            # Winner analysis
            winner = leaderboard_df.iloc[0]
            st.markdown(f'<div class="highlight-box">', unsafe_allow_html=True)
            st.markdown(f"**🥇 Best Performing Model: {winner['Model']}**")
            st.markdown(f"- Final Loss: {winner['Final Loss']:.4f}")
            st.markdown(f"- Architecture: {winner['Architecture']}")
            st.markdown(f"- Parameter Efficiency: {winner['Param Efficiency']:.6f} loss/M params")
            st.markdown('</div>', unsafe_allow_html=True)

    elif analysis_type == "📈 Loss Comparison":
        st.markdown('<h2 class="section-header">📈 Comprehensive Loss Analysis</h2>', unsafe_allow_html=True)
        
        if filtered_models:
            st.plotly_chart(create_comprehensive_loss_comparison(filtered_models), use_container_width=True)
            st.plotly_chart(create_training_efficiency_comparison(filtered_models), use_container_width=True)

    elif analysis_type == "🏗️ Architecture Analysis":
        st.markdown('<h2 class="section-header">🏗️ Architecture Comparison</h2>', unsafe_allow_html=True)
        
        # Model specs comparison
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Model Specifications")
            specs_df = pd.DataFrame(specs).T
            st.dataframe(specs_df, use_container_width=True)
        
        with col2:
            st.plotly_chart(create_model_architecture_radar(specs), use_container_width=True)

    elif analysis_type == "⚡ Training Efficiency":
        st.markdown('<h2 class="section-header">⚡ Training Efficiency Analysis</h2>', unsafe_allow_html=True)
        
        if filtered_models:
            st.plotly_chart(create_training_efficiency_comparison(filtered_models), use_container_width=True)

    elif analysis_type == "🔍 Pairwise Comparison":
        st.markdown('<h2 class="section-header">🔍 Detailed Pairwise Comparison</h2>', unsafe_allow_html=True)
        
        if len(selected_models) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                model_a = st.selectbox("Select First Model", selected_models, key="model_a")
            with col2:
                model_b = st.selectbox("Select Second Model", selected_models, key="model_b", index=1 if len(selected_models) > 1 else 0)
            
            if model_a != model_b:
                comparison_fig = create_pairwise_comparison(filtered_models, model_a, model_b)
                if comparison_fig:
                    st.plotly_chart(comparison_fig, use_container_width=True)
                    
                    # Statistical comparison
                    df_a = models_data[model_a]
                    df_b = models_data[model_b]
                    
                    if 'total_loss' in df_a.columns and 'total_loss' in df_b.columns:
                        min_len = min(len(df_a), len(df_b))
                        losses_a = df_a['total_loss'][:min_len]
                        losses_b = df_b['total_loss'][:min_len]
                        
                        # Statistical tests
                        correlation, p_value = pearsonr(losses_a, losses_b) if min_len > 1 else (0, 1)
                        t_stat, t_p = stats.ttest_ind(losses_a, losses_b) if min_len > 1 else (0, 1)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Correlation", f"{correlation:.3f}")
                        with col2:
                            st.metric("P-value", f"{p_value:.6f}")
                        with col3:
                            winner = model_a if losses_a.iloc[-1] < losses_b.iloc[-1] else model_b
                            st.metric("Winner", winner)
        else:
            st.info("Please select at least 2 models for pairwise comparison.")

    elif analysis_type == "📊 Statistical Analysis":
        st.markdown('<h2 class="section-header">📊 Advanced Statistical Analysis</h2>', unsafe_allow_html=True)
        
        if filtered_models:
            st.plotly_chart(create_advanced_statistical_analysis(filtered_models), use_container_width=True)

    elif analysis_type == "💡 Insights & Recommendations":
        st.markdown('<h2 class="section-header">💡 Key Insights & Recommendations</h2>', unsafe_allow_html=True)
        
        # Generate insights based on performance
        leaderboard_df = create_performance_leaderboard(filtered_models, specs)
        
        if not leaderboard_df.empty:
            best_model = leaderboard_df.iloc[0]
            insights = [
                f"🥇 **Best Overall Performance**: {best_model['Model']} with final loss of {best_model['Final Loss']:.4f}",
                f"⚡ **Most Parameter Efficient**: {leaderboard_df.loc[leaderboard_df['Param Efficiency'].idxmin(), 'Model']}",
                f"📈 **Fastest Improvement**: {leaderboard_df.loc[leaderboard_df['Improvement'].idxmax(), 'Model']}",
                f"🎯 **Most Stable Training**: {leaderboard_df.loc[leaderboard_df['Loss Std'].idxmin(), 'Model']}"
            ]
            
            for insight in insights:
                st.markdown(f'<div class="dark-insight-box">{insight}</div>', unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("### 🔧 Recommendations")
            recommendations = [
                "🚀 **For Production**: Use the model with lowest final loss and stable training",
                "💰 **For Cost Efficiency**: Consider parameter-efficient models (MoR variants)",
                "🔬 **For Research**: Explore hybrid architectures combining strengths",
                "⚖️ **For Balanced Use**: Consider trade-offs between performance and efficiency",
                "📊 **For Optimization**: Focus on learning rate schedules that work best"
            ]
            
            for rec in recommendations:
                st.markdown(f'<div class="highlight-box">{rec}</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #4ECDC4; padding: 20px;'>
            <p>🧠 Complete 4-Model Training Analysis Dashboard</p>
            <p>Built with Streamlit • Comprehensive model comparison and analysis</p>
            <p style='font-size: 0.9rem; opacity: 0.7;'>MoR 400M • Standard Transformer • Gemma MoR • Gemma Standard</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()