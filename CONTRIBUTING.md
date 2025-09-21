# Contributing to Portrait Wireframe Generator

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/portrait-wireframe-generator.git
   cd portrait-wireframe-generator
   ```

2. **Environment Setup**
   ```bash
   conda env create -f environment.yml
   conda activate portrait_outline
   ```

3. **Verify Installation**
   ```bash
   python setup.py
   ```

## 🔧 Development Workflow

### Making Changes

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow existing code style and conventions
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Changes**
   ```bash
   # Test individual components
   python image_processing/run_cutout.py --help
   python image_processing/wireframe_portrait_processor.py --help
   
   # Test complete pipeline
   python scripts/run_complete_pipeline.py --num-samples 1
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

### Code Style

- **Python**: Follow PEP 8 guidelines
- **Docstrings**: Use Google-style docstrings
- **Type hints**: Add type hints for function parameters and returns
- **Comments**: Write clear, concise comments for complex logic

### Example Code Style

```python
def process_portrait(image_path: str, output_dir: str) -> bool:
    """Process a portrait image through the wireframe pipeline.
    
    Args:
        image_path: Path to input portrait image
        output_dir: Directory to save output files
        
    Returns:
        True if processing successful, False otherwise
        
    Raises:
        FileNotFoundError: If input image doesn't exist
        ValueError: If image format not supported
    """
    # Implementation here
    pass
```

## 🐛 Bug Reports

When reporting bugs, please include:

- **System Information**: OS, Python version, GPU details
- **Environment**: Conda environment details (`conda list`)
- **Steps to Reproduce**: Clear reproduction steps
- **Expected vs Actual**: What you expected vs what happened
- **Error Messages**: Full error messages and stack traces
- **Sample Data**: If possible, provide sample images that cause issues

### Bug Report Template

```markdown
**System Information:**
- OS: [e.g., Ubuntu 22.04, macOS 14.0]
- Python: [e.g., 3.11.5]
- GPU: [e.g., AMD RX 7600, NVIDIA RTX 4080]

**Environment:**
```
conda list | grep -E "(mediapipe|opencv|onnx)"
```

**Bug Description:**
A clear description of the bug.

**Steps to Reproduce:**
1. Run command: `python ...`
2. Use input file: `...`
3. Observe error: `...`

**Expected Behavior:**
What should have happened.

**Error Output:**
```
Full error message here
```
```

## ✨ Feature Requests

For new features, please:

1. **Check Existing Issues**: Search for similar requests
2. **Describe Use Case**: Explain why the feature is needed
3. **Provide Examples**: Show how the feature would be used
4. **Consider Implementation**: Suggest how it might work

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test category
python -m pytest tests/test_segmentation.py -v
python -m pytest tests/test_wireframe.py -v

# Test with coverage
python -m pytest tests/ --cov=image_processing --cov-report=html
```

### Writing Tests

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test complete workflows
- **Performance Tests**: Verify processing speed and memory usage

Example test:

```python
import pytest
from image_processing.wireframe_portrait_processor import WireframeProcessor

def test_wireframe_generation():
    """Test basic wireframe generation."""
    processor = WireframeProcessor()
    result = processor.process_image("test_data/sample.jpg")
    
    assert result is not None
    assert result.has_construction_lines
    assert result.has_face_mesh
    assert len(result.landmarks) > 0
```

## 📚 Documentation

### Documentation Types

- **README.md**: Project overview and quick start
- **API Documentation**: Function and class documentation
- **Tutorials**: Step-by-step guides
- **Examples**: Code examples and use cases

### Writing Documentation

- Use clear, concise language
- Provide code examples
- Include expected outputs
- Keep documentation up-to-date with code changes

## 🔄 Pull Request Process

1. **Before Submitting**
   - Ensure all tests pass
   - Update documentation
   - Check code style compliance
   - Verify backward compatibility

2. **Pull Request Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement
   - [ ] Breaking change
   
   ## Testing
   - [ ] Added unit tests
   - [ ] Tested with sample data
   - [ ] Verified demo functionality
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] No breaking changes (or documented)
   ```

3. **Review Process**
   - Maintainers will review your PR
   - Address feedback and suggestions
   - Ensure CI checks pass
   - Merge when approved

## 🌟 Contributing Areas

We welcome contributions in these areas:

### Core Functionality
- **New Wireframe Types**: Additional wireframe generation methods
- **Performance Optimization**: Speed and memory improvements
- **GPU Support**: Enhanced GPU acceleration
- **Format Support**: Additional input/output formats

### User Interface
- **Web Demo Enhancements**: New features for the interactive demo
- **Command Line Tools**: Better CLI experience
- **Configuration**: More flexible configuration options

### Documentation
- **Tutorials**: Step-by-step guides for specific use cases
- **Examples**: More sample code and demonstrations
- **API Reference**: Comprehensive API documentation

### Testing
- **Test Coverage**: Increase test coverage
- **Performance Tests**: Benchmark and performance testing
- **Cross-platform Testing**: Ensure compatibility across systems

## 🎯 Release Process

### Version Numbering
We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version numbers updated
- [ ] Demo functionality verified

## 💬 Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussion
- **Pull Request Comments**: Code review and technical discussion

## 📝 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in:
- **README.md**: Contributors section
- **CHANGELOG.md**: Credit for specific changes
- **GitHub**: Contributor graph and statistics

Thank you for contributing! 🎨✨