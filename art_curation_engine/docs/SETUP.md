# 🛠️ Setup Guide

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: ~5GB for models and indices
- **API Access**: OpenAI API key

### Required Data Files
This curation engine requires specific data files that are **not included** in the repository due to size constraints:

#### 1. Research Database (`data/markdown/`)
- **Content**: 685 color psychology research document chunks
- **Format**: Markdown files with psychological research content
- **Purpose**: Evidence-based RAG brief generation

#### 2. Artwork Metadata (`metadata.jsonl`)
- **Content**: 298 artworks with detailed metadata
- **Format**: JSONL with fields like emotions, themes, colors, styles
- **Purpose**: Candidate filtering and collection

#### 3. CLIP Index (`indices/clip_faiss/`)
- **Content**: Pre-computed image embeddings
- **Files**: `faiss.index`, `id_map.json`
- **Purpose**: Semantic image-text similarity search

## Installation Steps

### 1. Clone and Setup
```bash
git clone <repository-url>
cd art_curation_engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
nano .env  # or use your preferred editor
```

### 4. Data Setup
```bash
# Create required directories
mkdir -p data/markdown
mkdir -p indices/clip_faiss

# Add your data files:
# - Place research documents in data/markdown/
# - Place metadata.jsonl in root directory
# - Place CLIP index files in indices/clip_faiss/
```

### 5. Verify Installation
```bash
python tests/test_step5_stagea_step6_integration.py
```

## API Keys Setup

### OpenAI API Key
1. Visit [OpenAI API](https://platform.openai.com/api-keys)
2. Create new API key
3. Add to `.env` file:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

### Optional: Fireworks API (Alternative LLM)
1. Visit [Fireworks AI](https://fireworks.ai/)
2. Get API key
3. Add to `.env` file:
   ```
   FIREWORKS_API_KEY=your-fireworks-key
   ```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# If you get import errors, ensure you're in the right directory
cd art_curation_engine
python -c "from core import RAGSessionBrief; print('✅ Imports working')"
```

#### 2. FAISS Installation Issues
```bash
# Try CPU version first
pip install faiss-cpu

# For Apple Silicon Macs
conda install -c conda-forge faiss-cpu
```

#### 3. Torch Installation Issues
```bash
# For CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 4. Memory Issues
- Reduce batch sizes in `.env`
- Use smaller embedding models
- Monitor RAM usage during testing

### Performance Optimization

#### 1. Enable GPU (if available)
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

#### 2. Model Caching
- First run downloads models (~2GB)
- Subsequent runs use cached models
- Cache location: `~/.cache/huggingface/`

#### 3. Batch Size Tuning
Adjust in `.env`:
```
STEP6_BATCH_SIZE=5  # Reduce if memory issues
STEP6_MAX_WORKERS=2  # Reduce for lower-end systems
```

## Data Requirements Detail

### Research Database Structure
```
data/markdown/
├── psychology_paper_001.md
├── psychology_paper_002.md
└── ...
```

### Metadata Format
```jsonl
{"id": 1, "title": "Sunset Lake", "emotions": ["peaceful", "calm"], "colors": ["orange", "blue"], ...}
{"id": 2, "title": "Abstract Motion", "emotions": ["energetic", "dynamic"], "colors": ["red", "yellow"], ...}
```

### CLIP Index Files
```
indices/clip_faiss/
├── faiss.index     # FAISS vector index
└── id_map.json     # ID to artwork mapping
```

## Next Steps

After setup completion:
1. Run the test suite: `python tests/run_step9_tests.py`
2. Try the integration test: `python tests/test_step5_stagea_step6_integration.py`
3. Review the main README.md for usage examples
4. Explore individual components in the `core/` directory