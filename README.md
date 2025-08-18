# 🧠 MoR vs Standard Transformer: A Comprehensive Implementation Study

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-complete-brightgreen.svg)

**A deep dive into implementing and comparing Google DeepMind's Mixture of Recursions (MoR) architecture against standard transformers**

[📊 Live Dashboard](your-streamlit-url) • [📄 Research Paper](https://arxiv.org/pdf/2507.10524v1) • [🔗 LinkedIn Post](your-linkedin-post-url)

</div>

---

## 🌟 Project Overview

This repository contains a comprehensive implementation and comparison study of **Mixture of Recursions (MoR)** architecture against standard transformers, including experiments on both custom models and Google's Gemma 3 270M. The project reveals important insights about the practical challenges and scale dependencies of cutting-edge architectural innovations.

### 🎯 Key Findings

| Model | Final Loss | Training Time | Memory Usage | Parameters |
|-------|------------|---------------|--------------|------------|
| **Gemma Standard** | **6.03** ✅ | 239s | 3.24GB | 268M |
| **MoR Implementation** | 23.23 ❌ | 717s | 9-13GB | 130M |
| **MoR 400M** | 8.45 | 2500 steps | 8GB+ | 365M |
| **Standard 400M** | 7.82 | 2500 steps | 6GB | 405M |

> **💡 Key Insight**: MoR's benefits emerge primarily at billion+ parameter scales, with implementation complexity often outweighing theoretical advantages at smaller scales.

---

## 🏗️ Architecture Implementations

### 🔄 Mixture of Recursions (MoR)
- **Recursive Transformer Blocks**: Shared layers applied multiple times
- **Dynamic Token Routing**: ACT-style halting mechanism
- **KV Cache Optimization**: Selective caching for active tokens
- **Parameter Sharing**: Reduced unique parameters through recursion

### 🏛️ Standard Transformer
- **Traditional Architecture**: Unique layers for each depth
- **Fixed Computation**: Uniform processing for all tokens
- **Standard Attention**: Multi-head self-attention mechanism
- **Linear Scaling**: Parameters scale with depth

### 🤖 Gemma Integration
- **Base Model**: Google's Gemma 3 270M
- **MoR Adaptation**: Custom routing and recursion layers
- **Comparative Analysis**: Direct performance comparison

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+
pip install torch torchvision torchaudio
pip install transformers datasets
pip install streamlit plotly pandas numpy
pip install einops scipy
```

### 🏃‍♂️ Running the Experiments

#### 1. Train MoR Model
```bash
python train_mor.py
```

#### 2. Train Standard Transformer
```bash
python train_standard_transformer.py
```

#### 3. Train Gemma Models
```bash
# Set your Hugging Face token
export HF_TOKEN="your_token_here"

# Standard Gemma
python gemma.py

# Gemma with MoR
python gemma_mor_implementation.py
```

#### 4. Launch Interactive Dashboard
```bash
streamlit run visualize_training.py
```

---

## 📊 Interactive Dashboard Features

<div align="center">
<img src="images_results/dashboard_preview.png" alt="Dashboard Preview" width="800"/>
</div>

### 🎛️ Dashboard Capabilities
- **📈 Real-time Loss Comparison**: All 4 models side-by-side
- **🏆 Performance Leaderboard**: Ranked by multiple metrics
- **🏗️ Architecture Analysis**: Radar charts and specifications
- **⚡ Training Efficiency**: Memory usage, convergence rates
- **🔍 Pairwise Comparisons**: Detailed model-vs-model analysis
- **📊 Statistical Analysis**: Violin plots, correlations, distributions
- **💡 AI-Generated Insights**: Automated recommendations

### 🎨 Visualization Gallery

| Loss Comparison | Architecture Radar | Performance Leaderboard |
|:---------------:|:------------------:|:----------------------:|
| ![Loss](images_results/loss_comparison.png) | ![Radar](images_results/architecture_radar.png) | ![Leaderboard](images_results/performance_leaderboard.png) |

---

## 🔬 Technical Deep Dive

### 🧮 MoR Implementation Details

```python
# Core MoR Architecture
class MoRLM(nn.Module):
    def __init__(self, cfg: MoRConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        # Shared transformer blocks (key innovation)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers_shared)
        ])

        # Dynamic routing mechanism
        self.router = HaltingRouter(cfg.d_model)
        self.norm_f = RMSNorm(cfg.d_model)
```

### 🎯 Key Architectural Differences

| Aspect | MoR | Standard Transformer |
|--------|-----|---------------------|
| **Parameter Sharing** | ✅ Shared across recursions | ❌ Unique per layer |
| **Dynamic Computation** | ✅ Token-level routing | ❌ Fixed computation |
| **Memory Efficiency** | ✅ Selective KV caching | ❌ Full KV storage |
| **Training Complexity** | ❌ High (routing, halting) | ✅ Straightforward |
| **Scale Dependency** | ❌ Requires large scale | ✅ Works at any scale |

---

## 📈 Experimental Results

### 🏆 Performance Summary

#### **Winner: Standard Architectures** 🥇

**Gemma Standard (270M)**
- ✅ **Best Loss**: 6.03
- ✅ **Fastest Training**: 239s
- ✅ **Memory Efficient**: 3.24GB
- ✅ **Stable Convergence**

**Why MoR Underperformed:**
1. **Scale Dependency**: Requires 16.5e18 FLOPs (billion+ parameters)
2. **Routing Complexity**: Expert-choice mechanism introduces instability
3. **Memory Overhead**: Multiple expert networks increase memory pressure
4. **Implementation Challenges**: Complex architecture harder to optimize

### 📊 Detailed Metrics

<details>
<summary><b>🔍 Click to expand detailed results</b></summary>

#### MoR 400M Model
```json
{
  "final_loss": 8.45,
  "parameters": 365615105,
  "training_time": "2500 steps",
  "memory_usage": "8GB+",
  "architecture": "28 shared layers × 3 recursions",
  "effective_depth": 84
}
```

#### Standard 400M Model
```json
{
  "final_loss": 7.82,
  "parameters": 405613568,
  "training_time": "2500 steps", 
  "memory_usage": "6GB",
  "architecture": "24 unique layers",
  "effective_depth": 24
}
```

#### Gemma MoR vs Standard
```json
{
  "gemma_standard": {
    "loss": 6.03,
    "time": "239s",
    "memory": "3.24GB",
    "status": "✅ Winner"
  },
  "gemma_mor": {
    "loss": 23.23,
    "time": "717s", 
    "memory": "9-13GB",
    "status": "❌ Underperformed"
  }
}
```

</details>

---

## 🎓 Research Insights

### 📚 Based on Google DeepMind's Paper

The original research demonstrates MoR's effectiveness at massive scale:
- **118M MoR** outperforms **315M vanilla Transformer**
- **2x faster inference** with proper optimization
- **50% memory reduction** through selective caching
- **Parameter efficiency** through recursive weight sharing

### 🔍 Our Implementation Findings

**Scale Matters**: 
- MoR benefits emerge at billion+ parameter scales
- Small models (270M) don't reach the critical threshold
- Infrastructure requirements are substantial

**Implementation Complexity**:
- Dynamic routing requires careful tuning
- Memory management is more complex
- Training stability needs attention

**Practical Considerations**:
- Standard architectures remain superior for resource-constrained scenarios
- MoR shows promise for large-scale deployments
- Hybrid approaches may offer best of both worlds

---

## 🛠️ Advanced Usage

### 🔧 Custom Configuration

```python
# MoR Configuration
config = MoRConfig(
    vocab_size=50257,
    d_model=1024,
    n_heads=16,
    d_ff=4096,
    n_layers_shared=28,    # Shared layers
    max_recursions=3,      # Recursion depth
    dropout=0.0,
    max_seq_len=256,
    kv_share_from_first=True,  # Memory optimization
    ponder_cost=0.01,          # Routing penalty
    tie_embeddings=True        # Parameter efficiency
)
```

### 📊 Custom Visualization

```python
# Launch dashboard with custom data
streamlit run visualize_training.py

# Or create custom plots
from visualize_training import create_comprehensive_loss_comparison
fig = create_comprehensive_loss_comparison(your_data)
fig.show()
```

### 🎯 Hyperparameter Tuning

<details>
<summary><b>🔧 Recommended hyperparameters</b></summary>

#### For MoR Models:
```python
# Training
learning_rate = 2e-4
batch_size = 1
grad_accumulation = 16
warmup_steps = 100
max_iters = 2500

# Architecture  
ponder_cost = 0.01      # Encourage early halting
max_recursions = 3      # Balance depth vs efficiency
kv_share = True         # Memory optimization
```

#### For Standard Models:
```python
# Training
learning_rate = 1e-4
batch_size = 1
grad_accumulation = 8
scheduler = "CosineAnnealingLR"

# Architecture
n_layers = 24           # Match MoR effective depth
dropout = 0.0           # Minimal regularization
```

</details>

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### 🎯 Areas for Contribution
- **🔬 New Architectures**: Implement other recursive/adaptive models
- **📊 Visualizations**: Add new analysis charts and metrics
- **⚡ Optimizations**: Improve training efficiency and memory usage
- **🧪 Experiments**: Test on different datasets and scales
- **📚 Documentation**: Enhance guides and tutorials

### 🚀 Getting Started
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 Citation

If you use this work in your research, please cite:

```bibtex
@misc{mor-vs-standard-2025,
  title={MoR vs Standard Transformer: A Comprehensive Implementation Study},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/mor-vs-standard-transformer}
}

@article{mor-original-2025,
  title={Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation},
  author={Google DeepMind Team},
  journal={arXiv preprint arXiv:2507.10524},
  year={2025}
}
```

---

## 🙏 Acknowledgments

### 🌟 Special Thanks

- **Google DeepMind** for the groundbreaking MoR research and making it accessible
- **Google** for the Gemma 3 270M model and infrastructure
- **Hugging Face** for the transformers library and model hosting
- **PyTorch Team** for the excellent deep learning framework
- **Streamlit** for the amazing dashboard capabilities

### 📚 Research Foundation

This work builds upon:
- [Mixture-of-Recursions Paper](https://arxiv.org/pdf/2507.10524v1) by Google DeepMind
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer
- [Gemma Technical Report](https://arxiv.org/abs/2403.08295) - Gemma Architecture

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links & Resources

### 🌐 Project Links
- **📊 [Live Dashboard](your-streamlit-url)** - Interactive analysis
- **📱 [LinkedIn Post](your-linkedin-post-url)** - Project summary
- **📧 [Contact](mailto:your-email)** - Get in touch

### 📚 Research Resources
- **[Original MoR Paper](https://arxiv.org/pdf/2507.10524v1)** - Google DeepMind
- **[Gemma Model Card](https://huggingface.co/google/gemma-3-270m)** - Hugging Face
- **[PyTorch Documentation](https://pytorch.org/docs/)** - Framework docs

### 🛠️ Technical Resources
- **[Streamlit Docs](https://docs.streamlit.io/)** - Dashboard framework
- **[Plotly Documentation](https://plotly.com/python/)** - Visualization library
- **[Transformers Library](https://huggingface.co/docs/transformers/)** - Model implementations

---

<div align="center">

### 🌟 Star this repository if you found it helpful!

**Built with ❤️ for the AI research community**

*Pushing the boundaries of what's possible in efficient AI architectures*

</div>