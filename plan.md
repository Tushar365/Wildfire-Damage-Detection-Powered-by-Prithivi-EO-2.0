# Wildfire Damage Detection with Prithivi 2.0

## Project Overview

A Flask-based application that uses Prithivi-EO-2.0 to detect wildfire damage by comparing pre and  
 post GeoTIFF images (6-band HLS format). Returns pre-image, post-image, and a change mask showing  
 damage areas.

## Architecture

### Technology Stack

- **Package Manager**: UV (fast Python package installer)
- **Backend**: Flask (simple web server)
- **Model**: Prithivi-EO-2.0-300M (base model, no fine-tuning)
- **Geospatial**: Rasterio for GeoTIFF handling
- **ML Framework**: PyTorch + TerraTorch
- **Runtime**: CPU-only inference  


### Project Structure

```
Changedetection-prithivi2.0/
├── pyproject.toml              # UV project configuration
├── README.md                   # Setup and usage instructions
├── app.py                      # Flask application
├── model/
│   ├── __init__.py
│   ├── prithivi_model.py       # Model loading and inference
│   └── processor.py            # GeoTIFF preprocessing
├── static/
│   ├── css/
│   │   └── style.css           # Simple styling
│   └── js/
│       └── main.js             # Frontend interactions
├── templates/
│   └── index.html              # Upload interface
└── uploads/                    # Temporary storage (gitignored)
```

## Implementation Plan

### Phase 1: Project Setup

1. Initialize UV project with `pyproject.toml`
2. Configure dependencies:

- Flask
- torch (CPU version)
- terratorch
- rasterio
- numpy
- pillow

3. Create directory structure
4. Add .gitignore (uploads/, model cache, **pycache**)  


### Phase 2: Model Integration

1. **model/prithivi_model.py**:

- Load Prithivi-EO-2.0-300M from Hugging Face
- Initialize model in CPU mode
- Implement inference function for change detection
- Cache model in memory (singleton pattern)  


2. **model/processor.py**:

- Load 6-band GeoTIFF files with rasterio
- Validate band count and format
- Normalize reflectance values
- Prepare input tensors for Prithivi
- Generate change mask from model output
- Export results as viewable images (PNG) and original GeoTIFF format  


### Phase 3: Flask Application

1. **app.py**:

- Single route `/` for interface
- POST `/detect` endpoint:
- Accept two GeoTIFF uploads (pre, post)
- Validate file formats
- Process through Prithivi model
- Return results (pre, post, mask images)
- Error handling for invalid inputs
- Progress indicators (optional but helpful for CPU)  


2. **templates/index.html**:

- Clean, simple upload form
- Two file inputs (pre-event, post-event)
- Submit button
- Results display area (3 images side-by-side)
- Loading indicator during processing  


3. **static/css/style.css**:

- Minimal, functional styling
- Responsive layout
- Image comparison view  


4. **static/js/main.js**:

- Form submission via fetch API
- Display uploaded images immediately
- Show loading state during processing
- Display results when ready  


### Phase 4: Change Detection Pipeline

The core algorithm flow:

1. Load pre and post GeoTIFFs (6 bands each)
2. Validate spatial alignment (same dimensions, CRS)
3. Stack as temporal sequence [pre, post]
4. Pass through Prithivi encoder
5. Use encoder embeddings to compute change
6. Generate binary mask (damaged/not damaged)
7. Apply threshold for visualization
8. Save outputs with geospatial metadata preserved  


### Phase 5: Testing & Documentation

1. Create sample test with small GeoTIFF files
2. Document API usage
3. Add README with:

- Installation with UV
- Running the server
- Expected input format
- Output interpretation
- Performance notes (CPU timing)  


## Critical Files

### 1. `pyproject.toml`

- UV configuration
- Python version (3.10+)
- Dependencies with versions  


### 2. `model/prithivi_model.py`

- Model initialization
- Change detection inference
- Key function: `detect_changes(pre_path, post_path) -> dict`  


### 3. `model/processor.py`

- GeoTIFF I/O
- Data preprocessing
- Output generation
- Key functions: `load_geotiff()`, `save_results()`  


### 4. `app.py`

- Flask routes
- File upload handling
- Integration with model  


### 5. `templates/index.html`

- User interface
- Results visualization  


## Key Design Decisions

### Why Prithivi-EO-2.0-300M (not 600M)?

- Smaller model = faster CPU inference
- Still highly accurate for change detection
- 300M parameters is the sweet spot for CPU  


### Why Base Model (No Fine-tuning)?

- Pre-trained on massive dataset including fire scars
- Designed for zero-shot change detection
- Saves time and computational resources  


### Why TerraTorch?

- Official framework for Prithivi models
- Handles complex preprocessing
- Provides utilities for geospatial ML  


### CPU Optimization Strategy

- Load model once at startup (singleton)
- Use PyTorch CPU optimizations
- Process one pair at a time
- Consider adding queue for multiple requests  


## Verification Plan

### End-to-End Test

1. Start Flask server: `uv run flask run`
2. Open browser to `http://localhost:5000`
3. Upload pre-event GeoTIFF (6 bands)
4. Upload post-event GeoTIFF (6 bands, same area)
5. Submit and wait for processing
6. Verify three outputs displayed:

- Pre-event image (RGB visualization)
- Post-event image (RGB visualization)
- Change mask (damaged areas highlighted)

7. Download results and verify GeoTIFF metadata preserved  


### Expected Behavior

- Valid 6-band HLS inputs → successful detection
- Mismatched dimensions → clear error message
- Wrong band count → validation error
- Processing time: 2-5 minutes on CPU (acceptable)  


## Sources

- [Prithvi-EO-2.0 GitHub Repository](https://github.com/NASA-IMPACT/Prithvi-EO-2.0)
- [Prithvi-EO-2.0-300M on Hugging  
  Face](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)
- [Prithvi-EO-2.0 Paper on arXiv](https://arxiv.org/abs/2412.02732)
- [IBM Research Blog on Prithvi 2.0](https://research.ibm.com/blog/prithvi2-geospatial)
