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

# Colors
MOR_COLOR = '#e74c3c'   # red
STD_COLOR = '#3498db'   # blue
POS_COLOR = 'rgba(39, 174, 96, 0.25)'    # greenish (MoR better)
NEG_COLOR = 'rgba(231, 76, 60, 0.25)'    # reddish (Std better)
MOR_FILL = 'rgba(231, 76, 60, 0.14)'
STD_FILL = 'rgba(52, 152, 219, 0.14)'

# Set page config
st.set_page_config(
    page_title="MoR vs Standard Transformer Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .section-header {
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    .highlight-box {
        background: #f8f9fa;
        border-left: 5px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .dark-insight-box {
        background: #2c3e50;
        color: #ecf0f1;
        border-left: 5px solid #3498db;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Load data function
@st.cache_data
def load_training_data():
    """Load and process training data from JSON files"""
    try:
        # Load MoR 400M data
        with open('training_metrics_400m.json', 'r') as f:
            mor_400m_data = json.load(f)

        # Load Standard data
        with open('training_metrics.json', 'r') as f:
            standard_data = json.load(f)

        # Load training loss data
        with open('standard_training_data.json', 'r') as f:
            standard_train_data = json.load(f)

        with open('mor_training_data.json', 'r') as f:
            mor_train_data = json.load(f)

        # Create DataFrames
        mor_400m_df = pd.DataFrame(mor_400m_data)
        standard_df = pd.DataFrame(standard_data)

        # Add model type
        mor_400m_df['model'] = 'MoR_400M'
        standard_df['model'] = 'Standard'

        # Create training loss DataFrames
        standard_train_df = pd.DataFrame(
            {
                'epoch': range(len(standard_train_data['train_losses'])),
                'train_loss': standard_train_data['train_losses'],
                'model': 'Standard',
            }
        )

        mor_train_df = pd.DataFrame(
            {
                'epoch': range(len(mor_train_data['train_losses'])),
                'train_loss': mor_train_data['train_losses'],
                'model': 'MoR',
            }
        )

        return mor_400m_df, standard_df, standard_train_df, mor_train_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

# Model architecture data
@st.cache_data
def get_model_specs():
    """Get model specifications"""
    return {
        'MoR_400M': {
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
        },
        'Standard': {
            'parameters': 405_613_568,
            'size_mb': 1547.3,
            'd_model': 1024,
            'n_heads': 16,
            'd_ff': 4096,
            'n_layers': 24,
            'effective_depth': 24,
            'architecture_type': 'Standard Transformer',
            'vocab_size': 50257,
        },
    }

def _first_max_lr_step(df):
    if 'lr' not in df.columns or 'steps' not in df.columns or len(df) == 0:
        return None
    mx = df['lr'].max()
    idx = df[df['lr'] >= mx * 0.9999].index
    if len(idx) == 0:
        return None
    return int(df.loc[idx[0], 'steps'])

def _min_loss_marker(df):
    if 'total_loss' not in df.columns or 'steps' not in df.columns or len(df) == 0:
        return None, None
    i = int(df['total_loss'].idxmin())
    return float(df.loc[i, 'steps']), float(df.loc[i, 'total_loss'])

def create_training_metrics_comparison(mor_df, std_df):
    """Create comprehensive training metrics comparison with key markers"""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            'Total Loss Comparison',
            'Language Model Loss',
            'Learning Rate Schedule',
            'Ponder Cost (MoR Only)',
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}], [{"secondary_y": False}, {"secondary_y": False}]],
    )

    # Total Loss
    fig.add_trace(
        go.Scatter(
            x=mor_df['steps'],
            y=mor_df['total_loss'],
            name='MoR 400M',
            line=dict(color=MOR_COLOR, width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=std_df['steps'],
            y=std_df['total_loss'],
            name='Standard',
            line=dict(color=STD_COLOR, width=2),
        ),
        row=1,
        col=1,
    )

    # Mark min loss points
    mor_min_step, mor_min_loss = _min_loss_marker(mor_df)
    std_min_step, std_min_loss = _min_loss_marker(std_df)
    if mor_min_step is not None:
        fig.add_trace(
            go.Scatter(
                x=[mor_min_step],
                y=[mor_min_loss],
                mode='markers+text',
                text=["MoR min"],
                textposition='top center',
                marker=dict(color=MOR_COLOR, size=10, symbol='star'),
                name='MoR min',
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    if std_min_step is not None:
        fig.add_trace(
            go.Scatter(
                x=[std_min_step],
                y=[std_min_loss],
                mode='markers+text',
                text=["Std min"],
                textposition='top center',
                marker=dict(color=STD_COLOR, size=10, symbol='star'),
                name='Std min',
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # LM Loss
    if 'lm_loss' in mor_df.columns and 'lm_loss' in std_df.columns:
        fig.add_trace(
            go.Scatter(
                x=mor_df['steps'],
                y=mor_df['lm_loss'],
                name='MoR 400M LM',
                line=dict(color=MOR_COLOR, width=2, dash='dot'),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=std_df['steps'],
                y=std_df['lm_loss'],
                name='Standard LM',
                line=dict(color=STD_COLOR, width=2, dash='dot'),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    else:
        fig.add_annotation(text="LM loss not available", xref="x2", yref="y2", x=0.5, y=0.5, showarrow=False)

    # Learning Rate
    if 'lr' in mor_df.columns:
        fig.add_trace(
            go.Scatter(
                x=mor_df['steps'],
                y=mor_df['lr'],
                name='MoR 400M LR',
                line=dict(color=MOR_COLOR, width=2),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    if 'lr' in std_df.columns:
        fig.add_trace(
            go.Scatter(
                x=std_df['steps'],
                y=std_df['lr'],
                name='Standard LR',
                line=dict(color=STD_COLOR, width=2),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Ponder Cost (MoR only)
    if 'ponder_cost' in mor_df.columns:
        fig.add_trace(
            go.Scatter(
                x=mor_df['steps'],
                y=mor_df['ponder_cost'],
                name='Ponder Cost',
                line=dict(color='#f39c12', width=2),
                showlegend=False,
            ),
            row=2,
            col=2,
        )
    else:
        fig.add_annotation(text="Ponder cost not available", xref="x4", yref="y4", x=0.5, y=0.5, showarrow=False)

    # Warmup markers on LR subplot
    mor_warm = _first_max_lr_step(mor_df)
    std_warm = _first_max_lr_step(std_df)
    if mor_warm is not None:
        fig.add_vline(x=mor_warm, line=dict(color=MOR_COLOR, dash='dash'), row=2, col=1)
    if std_warm is not None:
        fig.add_vline(x=std_warm, line=dict(color=STD_COLOR, dash='dash'), row=2, col=1)

    fig.update_layout(
        height=800,
        title_text="Training Metrics Comparison",
        title_x=0.5,
        showlegend=True,
        template="plotly_white",
    )

    return fig

def create_loss_distribution_analysis(mor_df, std_df):
    """Create loss distribution analysis"""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=('Loss Distribution', 'Loss Convergence Analysis'),
    )

    # Distribution plot
    fig.add_trace(
        go.Histogram(
            x=mor_df['total_loss'],
            name='MoR 400M',
            opacity=0.7,
            nbinsx=30,
            marker_color=MOR_COLOR,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=std_df['total_loss'],
            name='Standard',
            opacity=0.7,
            nbinsx=30,
            marker_color=STD_COLOR,
        ),
        row=1,
        col=1,
    )

    # Convergence analysis (moving average)
    window = 10
    mor_ma = mor_df['total_loss'].rolling(window=window).mean()
    std_ma = std_df['total_loss'].rolling(window=window).mean()

    fig.add_trace(
        go.Scatter(
            x=mor_df['steps'],
            y=mor_ma,
            name=f'MoR MA({window})',
            line=dict(color=MOR_COLOR, width=3),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=std_df['steps'],
            y=std_ma,
            name=f'Standard MA({window})',
            line=dict(color=STD_COLOR, width=3),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=500,
        title_text="Loss Distribution and Convergence Analysis",
        title_x=0.5,
        template="plotly_white",
    )

    return fig

def create_performance_radar_chart(specs):
    """Create radar chart comparing model performance"""
    categories = ['Parameters (M)', 'Model Size (GB)', 'Effective Depth', 'd_model', 'n_heads', 'd_ff/1000']

    mor_values = [
        specs['MoR_400M']['parameters'] / 1e6,
        specs['MoR_400M']['size_mb'] / 1000,
        specs['MoR_400M']['effective_depth'],
        specs['MoR_400M']['d_model'] / 100,
        specs['MoR_400M']['n_heads'],
        specs['MoR_400M']['d_ff'] / 1000,
    ]

    std_values = [
        specs['Standard']['parameters'] / 1e6,
        specs['Standard']['size_mb'] / 1000,
        specs['Standard']['effective_depth'],
        specs['Standard']['d_model'] / 100,
        specs['Standard']['n_heads'],
        specs['Standard']['d_ff'] / 1000,
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=mor_values,
            theta=categories,
            fill='toself',
            name='MoR 400M',
            line_color=MOR_COLOR,
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=std_values,
            theta=categories,
            fill='toself',
            name='Standard',
            line_color=STD_COLOR,
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(max(mor_values), max(std_values)) * 1.1])),
        showlegend=True,
        title="Model Architecture Comparison",
        title_x=0.5,
        height=500,
    )

    return fig

def create_training_efficiency_analysis(mor_train_df, std_train_df):
    """Create training efficiency analysis"""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            'Training Loss Curves',
            'Loss Improvement Rate',
            'Training Stability',
            'Convergence Speed',
        ),
    )

    # Training curves
    fig.add_trace(
        go.Scatter(
            x=mor_train_df['epoch'],
            y=mor_train_df['train_loss'],
            name='MoR Training',
            line=dict(color=MOR_COLOR, width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=std_train_df['epoch'],
            y=std_train_df['train_loss'],
            name='Standard Training',
            line=dict(color=STD_COLOR, width=2),
        ),
        row=1,
        col=1,
    )

    # Loss improvement rate (derivative)
    mor_diff = np.diff(mor_train_df['train_loss'])
    std_diff = np.diff(std_train_df['train_loss'])

    fig.add_trace(
        go.Scatter(
            x=mor_train_df['epoch'][1:],
            y=mor_diff,
            name='MoR Improvement',
            line=dict(color=MOR_COLOR, width=2),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=std_train_df['epoch'][1:],
            y=std_diff,
            name='Standard Improvement',
            line=dict(color=STD_COLOR, width=2),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Training stability (rolling std)
    window = 50
    mor_std = mor_train_df['train_loss'].rolling(window=window).std()
    std_std = std_train_df['train_loss'].rolling(window=window).std()

    fig.add_trace(
        go.Scatter(
            x=mor_train_df['epoch'],
            y=mor_std,
            name='MoR Stability',
            line=dict(color=MOR_COLOR, width=2),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=std_train_df['epoch'],
            y=std_std,
            name='Standard Stability',
            line=dict(color=STD_COLOR, width=2),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Convergence speed (time to reach certain thresholds)
    thresholds = [8, 6, 4, 2, 1]
    mor_convergence = []
    std_convergence = []

    for threshold in thresholds:
        mor_idx = mor_train_df[mor_train_df['train_loss'] <= threshold].index
        std_idx = std_train_df[std_train_df['train_loss'] <= threshold].index

        mor_convergence.append(mor_idx[0] if len(mor_idx) > 0 else len(mor_train_df))
        std_convergence.append(std_idx[0] if len(std_idx) > 0 else len(std_train_df))

    fig.add_trace(
        go.Bar(
            x=[f"Loss<{t}" for t in thresholds],
            y=mor_convergence,
            name='MoR Convergence',
            marker_color=MOR_COLOR,
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=[f"Loss<{t}" for t in thresholds],
            y=std_convergence,
            name='Standard Convergence',
            marker_color=STD_COLOR,
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=800,
        title_text="Training Efficiency Analysis",
        title_x=0.5,
        template="plotly_white",
    )

    return fig

def create_statistical_analysis(mor_df, std_df):
    """Create statistical analysis of model performance"""
    mor_stats = {
        'mean_loss': mor_df['total_loss'].mean(),
        'std_loss': mor_df['total_loss'].std(),
        'min_loss': mor_df['total_loss'].min(),
        'final_loss': mor_df['total_loss'].iloc[-1],
        'improvement': mor_df['total_loss'].iloc[0] - mor_df['total_loss'].iloc[-1],
        'cv': mor_df['total_loss'].std() / mor_df['total_loss'].mean(),
    }

    std_stats = {
        'mean_loss': std_df['total_loss'].mean(),
        'std_loss': std_df['total_loss'].std(),
        'min_loss': std_df['total_loss'].min(),
        'final_loss': std_df['total_loss'].iloc[-1],
        'improvement': std_df['total_loss'].iloc[0] - std_df['total_loss'].iloc[-1],
        'cv': std_df['total_loss'].std() / std_df['total_loss'].mean(),
    }

    metrics = list(mor_stats.keys())
    mor_values = list(mor_stats.values())
    std_values = list(std_stats.values())

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[m.replace('_', ' ').title() for m in metrics],
            y=mor_values,
            name='MoR 400M',
            marker_color=MOR_COLOR,
        )
    )
    fig.add_trace(
        go.Bar(
            x=[m.replace('_', ' ').title() for m in metrics],
            y=std_values,
            name='Standard',
            marker_color=STD_COLOR,
        )
    )

    fig.update_layout(
        title="Statistical Performance Comparison",
        title_x=0.5,
        xaxis_title="Metrics",
        yaxis_title="Values",
        barmode='group',
        height=500,
        template="plotly_white",
    )

    return fig, mor_stats, std_stats

def create_memory_efficiency_chart(mor_df, std_df, specs):
    """Create memory efficiency analysis"""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=('Memory Usage Over Time', 'Parameter Efficiency'),
    )

    # Memory usage (only if column exists)
    if 'memory_gb' in mor_df.columns:
        fig.add_trace(
            go.Scatter(
                x=mor_df['steps'],
                y=mor_df['memory_gb'],
                name='MoR 400M Memory',
                line=dict(color=MOR_COLOR, width=2),
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_annotation(text="Memory usage not available", xref="x1", yref="y1", x=0.5, y=0.5, showarrow=False)

    # Parameter efficiency (performance per parameter)
    mor_efficiency = mor_df['total_loss'].iloc[-1] / (specs['MoR_400M']['parameters'] / 1e6)
    std_efficiency = std_df['total_loss'].iloc[-1] / (specs['Standard']['parameters'] / 1e6)

    fig.add_trace(
        go.Bar(
            x=['MoR 400M', 'Standard'],
            y=[mor_efficiency, std_efficiency],
            name='Loss per Million Parameters',
            marker_color=[MOR_COLOR, STD_COLOR],
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=500,
        title_text="Memory and Parameter Efficiency",
        title_x=0.5,
        template="plotly_white",
    )

    return fig

# New: helper to align both runs on steps
def align_on_steps(mor_df, std_df):
    cols = ['steps', 'total_loss', 'lm_loss', 'lr']
    use_cols_mor = [c for c in cols if c in mor_df.columns] + ['steps']
    use_cols_std = [c for c in cols if c in std_df.columns] + ['steps']
    df = pd.merge(
        mor_df[list(set(use_cols_mor))],
        std_df[list(set(use_cols_std))],
        on='steps',
        how='inner',
        suffixes=('_mor', '_std'),
    ).sort_values('steps')
    return df

# New: smoothed loss with variability bands
def create_smoothed_loss_bands(mor_df, std_df, window=25, sigma=1.0):
    fig = go.Figure()
    for label, df, color, fill in [
        ('MoR 400M', mor_df, MOR_COLOR, MOR_FILL),
        ('Standard', std_df, STD_COLOR, STD_FILL),
    ]:
        s = df['total_loss']
        ma = s.rolling(window=window, min_periods=max(2, window // 2)).mean()
        sd = s.rolling(window=window, min_periods=max(2, window // 2)).std()
        upper = ma + sigma * sd
        lower = ma - sigma * sd

        # CI fill
        fig.add_trace(
            go.Scatter(
                x=df['steps'],
                y=lower,
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False,
                hoverinfo='skip',
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df['steps'],
                y=upper,
                fill='tonexty',
                fillcolor=fill,
                line=dict(color='rgba(0,0,0,0)'),
                name=f'{label} variability (±{sigma:.1f}σ)',
            )
        )

        # Smoothed line
        fig.add_trace(
            go.Scatter(
                x=df['steps'],
                y=ma,
                name=f'{label} (smoothed)',
                line=dict(color=color, width=3),
            )
        )

    fig.update_layout(
        title=f"Smoothed Total Loss with Variability Bands (window={window})",
        title_x=0.5,
        template='plotly_white',
        height=500,
        yaxis_title="Total Loss",
        xaxis_title="Steps",
        legend_title="Legend",
    )
    return fig

# New: difference area plot (MoR - Standard); green below zero means MoR better
def create_loss_difference_area(mor_df, std_df, window=15):
    aligned = align_on_steps(mor_df, std_df)
    if len(aligned) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No overlapping steps to compare", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_white", height=420, title="MoR − Standard Loss Difference")
        return fig

    diff = aligned['total_loss_mor'] - aligned['total_loss_std']
    diff_sm = pd.Series(diff).rolling(window=window, min_periods=max(2, window // 2)).mean()

    pos = np.where(diff_sm >= 0, diff_sm, None)
    neg = np.where(diff_sm < 0, diff_sm, None)

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color='black', width=1, dash='dash'))

    fig.add_trace(
        go.Scatter(
            x=aligned['steps'],
            y=pos,
            name='Std better',
            fill='tozeroy',
            fillcolor=NEG_COLOR,
            line=dict(color='rgba(0,0,0,0)'),
            hovertemplate='Step %{x}<br>Diff (MoR-Std): %{y:.4f}<extra></extra>',
        )
    )
    fig.add_trace(
        go.Scatter(
            x=aligned['steps'],
            y=neg,
            name='MoR better',
            fill='tozeroy',
            fillcolor=POS_COLOR,
            line=dict(color='rgba(0,0,0,0)'),
            hovertemplate='Step %{x}<br>Diff (MoR-Std): %{y:.4f}<extra></extra>',
        )
    )
    fig.add_trace(
        go.Scatter(
            x=aligned['steps'],
            y=diff_sm,
            name='Difference (smoothed)',
            line=dict(color='#2c3e50', width=2),
        )
    )

    fig.update_layout(
        title=f"MoR − Standard Loss Difference (smoothed, window={window})",
        title_x=0.5,
        template='plotly_white',
        height=420,
        yaxis_title="Difference (MoR - Std)",
        xaxis_title="Steps",
    )
    return fig

# New: joint density of losses (Std vs MoR) on overlapping steps
def create_joint_density(aligned):
    if len(aligned) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No overlapping steps to compare", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_white", height=500, title="Joint Density of Losses")
        return fig

    fig = px.density_heatmap(
        aligned,
        x='total_loss_std',
        y='total_loss_mor',
        nbinsx=40,
        nbinsy=40,
        color_continuous_scale='Magma',
        title='Joint Density: Standard (x) vs MoR (y) Loss',
        labels={'total_loss_std': 'Standard Loss', 'total_loss_mor': 'MoR Loss'},
    )
    # Diagonal reference
    x_min = float(aligned['total_loss_std'].min())
    x_max = float(aligned['total_loss_std'].max())
    y_min = float(aligned['total_loss_mor'].min())
    y_max = float(aligned['total_loss_mor'].max())
    lo = min(x_min, y_min)
    hi = max(x_max, y_max)
    fig.add_shape(
        type="line",
        x0=lo, y0=lo, x1=hi, y1=hi,
        line=dict(color="cyan", dash="dash"),
    )
    fig.update_layout(template="plotly_white", height=500)
    return fig

# New: violin distributions
def create_violin_distributions(mor_df, std_df):
    df_long = pd.DataFrame({
        'loss': pd.concat([mor_df['total_loss'], std_df['total_loss']], ignore_index=True),
        'model': ['MoR 400M'] * len(mor_df) + ['Standard'] * len(std_df)
    })
    fig = px.violin(
        df_long,
        x='model',
        y='loss',
        color='model',
        color_discrete_map={'MoR 400M': MOR_COLOR, 'Standard': STD_COLOR},
        box=True,
        points='outliers',
        title='Loss Distributions by Model',
    )
    fig.update_layout(template='plotly_white', height=500, showlegend=False)
    return fig

# New: ECDF comparison
def create_ecdf(mor_df, std_df):
    df_long = pd.DataFrame({
        'loss': pd.concat([mor_df['total_loss'], std_df['total_loss']], ignore_index=True),
        'model': ['MoR 400M'] * len(mor_df) + ['Standard'] * len(std_df)
    })
    fig = px.ecdf(
        df_long,
        x='loss',
        color='model',
        color_discrete_map={'MoR 400M': MOR_COLOR, 'Standard': STD_COLOR},
        title='ECDF of Total Loss (left/steeper = better)',
    )
    fig.update_layout(template='plotly_white', height=420)
    return fig

# New: LR-colored scatter with smoothed overlay
def create_lr_vs_loss_scatter(mor_df, std_df, window=20):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('MoR 400M', 'Standard'))

    for col_idx, (label, df, color, scale) in enumerate([
        ('MoR 400M', mor_df, MOR_COLOR, 'Turbo'),
        ('Standard', std_df, STD_COLOR, 'Viridis'),
    ], start=1):
        scatter = go.Scatter(
            x=df['steps'],
            y=df['total_loss'],
            mode='markers',
            marker=dict(
                color=df['lr'] if 'lr' in df.columns else None,
                colorscale=scale,
                showscale=True,
                size=5,
                colorbar=dict(title='LR'),
            ),
            name=f'{label} (colored by LR)',
        )
        fig.add_trace(scatter, row=1, col=col_idx)

        ma = df['total_loss'].rolling(window=window, min_periods=max(2, window // 2)).mean()
        fig.add_trace(
            go.Scatter(
                x=df['steps'],
                y=ma,
                name=f'{label} (smoothed)',
                line=dict(color=color, width=3),
                showlegend=False,
            ),
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        title=f"Loss vs Steps Colored by Learning Rate (window={window})",
        title_x=0.5,
        template='plotly_white',
        height=520,
    )
    return fig

# New: Ponder cost panels (only if available)
def create_ponder_panels(mor_df):
    if 'ponder_cost' not in mor_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Ponder cost not available", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_white", height=420, title="Ponder Analysis")
        return fig

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Ponder Cost over Steps', 'Ponder vs Loss (MoR)'))
    fig.add_trace(
        go.Scatter(
            x=mor_df['steps'],
            y=mor_df['ponder_cost'],
            name='Ponder Cost',
            line=dict(color='#f39c12', width=2),
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=mor_df['ponder_cost'],
            y=mor_df['total_loss'],
            mode='markers',
            marker=dict(color='#f39c12', size=5, opacity=0.7),
            name='Ponder vs Loss',
        ),
        row=1, col=2
    )
    fig.update_xaxes(title_text="Steps", row=1, col=1)
    fig.update_yaxes(title_text="Ponder Cost", row=1, col=1)
    fig.update_xaxes(title_text="Ponder Cost", row=1, col=2)
    fig.update_yaxes(title_text="Total Loss", row=1, col=2)
    fig.update_layout(template='plotly_white', height=500, title="Ponder Cost Analysis (MoR)")
    return fig

def main():
    # Header
    st.markdown(
        '<h1 class="main-header">🧠 MoR vs Standard Transformer Analysis Dashboard</h1>',
        unsafe_allow_html=True,
    )

    # Load data
    mor_400m_df, standard_df, standard_train_df, mor_train_df = load_training_data()
    specs = get_model_specs()

    if mor_400m_df is None:
        st.error("Failed to load data. Please ensure all JSON files are available.")
        return

    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    st.sidebar.markdown("---")

    smoothing_window = st.sidebar.slider("Smoothing window (steps)", 5, 100, 25, help="Used in smoothed charts and bands")
    sigma_bands = st.sidebar.slider("Variability band (σ)", 0.5, 2.5, 1.0, 0.1, help="Controls shaded band width around smoothed curves")

    selected_analysis = st.sidebar.selectbox(
        "Select Analysis Type",
        [
            "Overview",
            "Training Metrics",
            "Architecture Comparison",
            "Performance Analysis",
            "Statistical Analysis",
            "Efficiency Analysis",
            "Detailed Insights",
        ],
    )

    if selected_analysis == "Overview":
        st.markdown('<h2 class="section-header">📈 Model Overview</h2>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
<div class="metric-card">
    <h3>MoR 400M</h3>
    <p><strong>{specs['MoR_400M']['parameters']:,}</strong> parameters</p>
    <p><strong>{specs['MoR_400M']['size_mb']:.1f}</strong> MB</p>
</div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
<div class="metric-card">
    <h3>Standard</h3>
    <p><strong>{specs['Standard']['parameters']:,}</strong> parameters</p>
    <p><strong>{specs['Standard']['size_mb']:.1f}</strong> MB</p>
</div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            final_mor_loss = float(mor_400m_df['total_loss'].iloc[-1])
            final_std_loss = float(standard_df['total_loss'].iloc[-1])
            st.markdown(
                f"""
<div class="metric-card">
    <h3>Final Loss</h3>
    <p>MoR: <strong>{final_mor_loss:.3f}</strong></p>
    <p>Std: <strong>{final_std_loss:.3f}</strong></p>
</div>
                """,
                unsafe_allow_html=True,
            )

        with col4:
            winner = "Standard" if final_std_loss < final_mor_loss else "MoR 400M"
            improvement = abs(final_std_loss - final_mor_loss) / max(final_std_loss, final_mor_loss) * 100
            st.markdown(
                f"""
<div class="metric-card">
    <h3>Winner</h3>
    <p><strong>{winner}</strong></p>
    <p>{improvement:.1f}% better</p>
</div>
                """,
                unsafe_allow_html=True,
            )

        st.plotly_chart(create_training_metrics_comparison(mor_400m_df, standard_df), use_container_width=True)
        st.plotly_chart(create_smoothed_loss_bands(mor_400m_df, standard_df, window=smoothing_window, sigma=sigma_bands), use_container_width=True)

    elif selected_analysis == "Training Metrics":
        st.markdown('<h2 class="section-header">📊 Training Metrics Analysis</h2>', unsafe_allow_html=True)
        st.plotly_chart(create_smoothed_loss_bands(mor_400m_df, standard_df, window=smoothing_window, sigma=sigma_bands), use_container_width=True)
        st.plotly_chart(create_loss_difference_area(mor_400m_df, standard_df, window=max(5, smoothing_window // 2)), use_container_width=True)
        st.plotly_chart(create_lr_vs_loss_scatter(mor_400m_df, standard_df, window=max(5, smoothing_window // 2)), use_container_width=True)
        st.plotly_chart(create_loss_distribution_analysis(mor_400m_df, standard_df), use_container_width=True)
        st.plotly_chart(create_ponder_panels(mor_400m_df), use_container_width=True)
        if standard_train_df is not None and mor_train_df is not None:
            st.plotly_chart(create_training_efficiency_analysis(mor_train_df, standard_train_df), use_container_width=True)

    elif selected_analysis == "Architecture Comparison":
        st.markdown('<h2 class="section-header">🏗️ Architecture Comparison</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### MoR 400M Architecture")
            st.json(specs['MoR_400M'])
        with col2:
            st.markdown("### Standard Architecture")
            st.json(specs['Standard'])
        st.plotly_chart(create_performance_radar_chart(specs), use_container_width=True)
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown(
            """
Key architectural differences: MoR uses recursive layers with shared parameters for an effective depth of 36 versus 24 in the Standard transformer. MoR trades additional compute for parameter efficiency, while Standard relies on independent layers with more parameters overall.
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

    elif selected_analysis == "Performance Analysis":
        st.markdown('<h2 class="section-header">⚡ Performance Analysis</h2>', unsafe_allow_html=True)
        stats_fig, mor_stats, std_stats = create_statistical_analysis(mor_400m_df, standard_df)
        st.plotly_chart(stats_fig, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### MoR 400M Statistics")
            stats_df = pd.DataFrame(list(mor_stats.items()), columns=['Metric', 'Value'])
            st.dataframe(stats_df, use_container_width=True)
        with col2:
            st.markdown("### Standard Statistics")
            stats_df = pd.DataFrame(list(std_stats.items()), columns=['Metric', 'Value'])
            st.dataframe(stats_df, use_container_width=True)

    elif selected_analysis == "Statistical Analysis":
        st.markdown('<h2 class="section-header">📈 Statistical Analysis</h2>', unsafe_allow_html=True)
        st.markdown("### Loss Correlation Analysis")
        min_len = min(len(mor_400m_df), len(standard_df))
        mor_losses = mor_400m_df['total_loss'][:min_len]
        std_losses = standard_df['total_loss'][:min_len]
        correlation, p_value = pearsonr(mor_losses, std_losses)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Correlation Coefficient", f"{correlation:.3f}")
        with col2:
            st.metric("P-value", f"{p_value:.6f}")
        with col3:
            significance = "Significant" if p_value < 0.05 else "Not Significant"
            st.metric("Statistical Significance", significance)
        fig = px.scatter(
            x=mor_losses,
            y=std_losses,
            labels={'x': 'MoR 400M Loss', 'y': 'Standard Loss'},
            title="Loss Correlation Between Models",
        )
        fig.add_shape(
            type="line",
            x0=float(mor_losses.min()),
            y0=float(std_losses.min()),
            x1=float(mor_losses.max()),
            y1=float(std_losses.max()),
            line=dict(color="red", dash="dash"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # T-test and effect size
        st.markdown("### Statistical Tests")
        t_stat, t_p_value = stats.ttest_ind(mor_losses, std_losses)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**T-Test Results**")
            st.write(f"T-statistic: {t_stat:.3f}")
            st.write(f"P-value: {t_p_value:.6f}")
            st.write(f"Significant difference: {'Yes' if t_p_value < 0.05 else 'No'}")
        with col2:
            st.markdown("**Effect Size (Cohen's d)**")
            pooled_std = np.sqrt(
                ((len(mor_losses) - 1) * np.var(mor_losses) + (len(std_losses) - 1) * np.var(std_losses))
                / (len(mor_losses) + len(std_losses) - 2)
            )
            cohens_d = (np.mean(mor_losses) - np.mean(std_losses)) / pooled_std
            st.write(f"Cohen's d: {cohens_d:.3f}")
        effect_size = "Small" if abs(cohens_d) < 0.2 else ("Medium" if abs(cohens_d) < 0.8 else "Large")
        st.write(f"Effect size: {effect_size}")

        # Advanced visuals
        st.markdown("### Distributional and Joint Views")
        aligned = align_on_steps(mor_400m_df, standard_df)
        st.plotly_chart(create_joint_density(aligned), use_container_width=True)
        st.plotly_chart(create_violin_distributions(mor_400m_df, standard_df), use_container_width=True)
        st.plotly_chart(create_ecdf(mor_400m_df, standard_df), use_container_width=True)

    elif selected_analysis == "Efficiency Analysis":
        st.markdown('<h2 class="section-header">⚙️ Efficiency Analysis</h2>', unsafe_allow_html=True)
        st.plotly_chart(create_memory_efficiency_chart(mor_400m_df, standard_df, specs), use_container_width=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### Parameter Efficiency")
            mor_param_eff = mor_400m_df['total_loss'].iloc[-1] / (specs['MoR_400M']['parameters'] / 1e6)
            std_param_eff = standard_df['total_loss'].iloc[-1] / (specs['Standard']['parameters'] / 1e6)
            st.metric("MoR 400M", f"{mor_param_eff:.4f}", "loss/M params")
            st.metric("Standard", f"{std_param_eff:.4f}", "loss/M params")
        with col2:
            st.markdown("### Memory Efficiency")
            mor_mem_eff = mor_400m_df['total_loss'].iloc[-1] / specs['MoR_400M']['size_mb']
            std_mem_eff = standard_df['total_loss'].iloc[-1] / specs['Standard']['size_mb']
            st.metric("MoR 400M", f"{mor_mem_eff:.4f}", "loss/MB")
            st.metric("Standard", f"{std_mem_eff:.4f}", "loss/MB")
        with col3:
            st.markdown("### Training Efficiency")
            mor_train_eff = (mor_400m_df['total_loss'].iloc[0] - mor_400m_df['total_loss'].iloc[-1]) / len(mor_400m_df)
            std_train_eff = (standard_df['total_loss'].iloc[0] - standard_df['total_loss'].iloc[-1]) / len(standard_df)
            st.metric("MoR 400M", f"{mor_train_eff:.4f}", "improvement/step")
            st.metric("Standard", f"{std_train_eff:.4f}", "improvement/step")

    elif selected_analysis == "Detailed Insights":
        st.markdown('<h2 class="section-header">🔍 Detailed Insights</h2>', unsafe_allow_html=True)
        st.markdown("### 🎯 Key Findings")
        final_mor_loss = mor_400m_df['total_loss'].iloc[-1]
        final_std_loss = standard_df['total_loss'].iloc[-1]
        insights = []
        if final_std_loss < final_mor_loss:
            improvement = (final_mor_loss - final_std_loss) / final_mor_loss * 100
            insights.append(f"✅ Standard Transformer outperforms MoR by {improvement:.1f}% in final loss")
        else:
            improvement = (final_std_loss - final_mor_loss) / final_std_loss * 100
            insights.append(f"✅ MoR outperforms Standard Transformer by {improvement:.1f}% in final loss")
        mor_param_ratio = specs['MoR_400M']['parameters'] / specs['Standard']['parameters']
        if mor_param_ratio < 1:
            param_savings = (1 - mor_param_ratio) * 100
            insights.append(f"💡 MoR uses {param_savings:.1f}% fewer parameters than Standard")
        else:
            param_excess = (mor_param_ratio - 1) * 100
            insights.append(f"💡 MoR uses {param_excess:.1f}% more parameters than Standard")
        mor_cv = mor_400m_df['total_loss'].std() / mor_400m_df['total_loss'].mean()
        std_cv = standard_df['total_loss'].std() / standard_df['total_loss'].mean()
        if mor_cv < std_cv:
            insights.append("📈 MoR shows more stable training (lower coefficient of variation)")
        else:
            insights.append("📈 Standard shows more stable training (lower coefficient of variation)")
        mor_improvement = mor_400m_df['total_loss'].iloc[0] - mor_400m_df['total_loss'].iloc[-1]
        std_improvement = standard_df['total_loss'].iloc[0] - standard_df['total_loss'].iloc[-1]
        if mor_improvement > std_improvement:
            insights.append("🚀 MoR shows faster convergence (greater total loss reduction)")
        else:
            insights.append("🚀 Standard shows faster convergence (greater total loss reduction)")
        for insight in insights:
            st.markdown(f'<div class="dark-insight-box">{insight}</div>', unsafe_allow_html=True)

        st.markdown("### 💡 Recommendations")
        recommendations = [
            "🔧 For production: consider the model with the lower final loss and tighter variability bands",
            "⚡ For resource-constrained environments: MoR can offer parameter efficiency with competitive loss",
            "🧪 For research: explore hybrid approaches combining recursion with standard layers",
            "📊 For optimization: tune learning rate schedule based on LR-colored scatter trends",
            "🎯 For downstream tasks: validate both architectures on task-specific metrics",
        ]
        for rec in recommendations:
            st.markdown(f'<div class="dark-insight-box">{rec}</div>', unsafe_allow_html=True)

        st.markdown("### 📥 Data Export")
        col1, col2 = st.columns(2)
        with col1:
            csv = mor_400m_df.to_csv(index=False)
            st.download_button(
                label="Download MoR CSV",
                data=csv,
                file_name="mor_400m_data.csv",
                mime="text/csv",
            )
        with col2:
            csv = standard_df.to_csv(index=False)
            st.download_button(
                label="Download Standard CSV",
                data=csv,
                file_name="standard_data.csv",
                mime="text/csv",
            )

    st.markdown("---")
    st.markdown(
        """
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🧠 MoR vs Standard Transformer Analysis Dashboard</p>
    <p>Built with Streamlit • Data-driven insights for model comparison</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    # Final guidance line on how to understand the visuals
    st.info(
        "How to read this: lower loss lines are better; wider shaded bands indicate more variability; green regions below zero in the MoR−Std difference chart mean MoR is ahead (red above zero favors Standard); left-shifted ECDF and tighter violin plots indicate better and more consistent performance; LR-colored scatter helps relate training phases (warmup/plateau/decay) to loss."
    )

if __name__ == "__main__":
    main()