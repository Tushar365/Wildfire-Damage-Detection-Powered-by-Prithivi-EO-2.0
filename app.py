
"""
Flask application for wildfire damage detection using Prithivi-EO-2.0.
"""

import os
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import traceback
from datetime import datetime

from model.prithivi_model import PrithiviChangeDetector
from model.processor import GeoTIFFProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'uploads/results'

# Allowed extensions
ALLOWED_EXTENSIONS = {'tif', 'tiff'}

# Initialize model and processor (lazy loading)
detector = None
processor = None


def get_detector():
    """Get or create model detector instance."""
    global detector
    if detector is None:
        logger.info("Initializing Prithivi model...")
        detector = PrithiviChangeDetector()
        logger.info("Model ready!")
    return detector


def get_processor():
    """Get or create GeoTIFF processor instance."""
    global processor
    if processor is None:
        processor = GeoTIFFProcessor()
    return processor


def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint."""
    try:
        model_info = get_detector().get_model_info()
        return jsonify({
            'status': 'healthy',
            'model': model_info
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/detect', methods=['POST'])
def detect():
    """
    Main detection endpoint.

    Expects:
        - pre_image: GeoTIFF file (6 bands, HLS format)
        - post_image: GeoTIFF file (6 bands, HLS format)
        - threshold: (optional) Detection threshold 0-1, default 0.5

    Returns:
        JSON with paths to result images
    """
    try:
        # Validate request
        if 'pre_image' not in request.files or 'post_image' not in request.files:
            return jsonify({
                'error': 'Both pre_image and post_image are required'
            }), 400

        pre_file = request.files['pre_image']
        post_file = request.files['post_image']

        if pre_file.filename == '' or post_file.filename == '':
            return jsonify({
                'error': 'Empty filename'
            }), 400

        if not (allowed_file(pre_file.filename) and allowed_file(post_file.filename)):
            return jsonify({
                'error': f'Only {ALLOWED_EXTENSIONS} files are allowed'
            }), 400

        # Get optional threshold parameter
        threshold = float(request.form.get('threshold', 0.5))
        if not 0 <= threshold <= 1:
            return jsonify({
                'error': 'Threshold must be between 0 and 1'
            }), 400

        logger.info(f"Processing detection request with threshold={threshold}")

        # Save uploaded files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pre_filename = secure_filename(f'{timestamp}_pre_{pre_file.filename}')
        post_filename = secure_filename(f'{timestamp}_post_{post_file.filename}')

        pre_path = os.path.join(app.config['UPLOAD_FOLDER'], pre_filename)
        post_path = os.path.join(app.config['UPLOAD_FOLDER'], post_filename)

        pre_file.save(pre_path)
        post_file.save(post_path)

        logger.info(f"Saved files: {pre_path}, {post_path}")

        # Process files
        proc = get_processor()

        # Load GeoTIFFs
        logger.info("Loading GeoTIFF files...")
        pre_data = proc.load_geotiff(pre_path)
        post_data = proc.load_geotiff(post_path)

        # Validate alignment
        logger.info("Validating spatial alignment...")
        proc.validate_alignment(pre_data, post_data)

        # Prepare input for model
        logger.info("Preparing model input...")
        pre_array = proc.normalize_reflectance(pre_data['data'])
        post_array = proc.normalize_reflectance(post_data['data'])

        # Run burn area detection (instead of generic change detection)
        logger.info("Running burn area detection...")
        det = get_detector()
        results = det.detect_burn_areas(
            pre_array,
            post_array,
            threshold=threshold,
            use_hybrid=True  # Fuse dNBR + Prithivi embeddings
        )

        # Create visualizations
        logger.info("Creating visualizations...")
        pre_rgb = proc.create_rgb_visualization(pre_data['data'])
        post_rgb = proc.create_rgb_visualization(post_data['data'])
        
        # Use fire colormap for burn visualization
        mask_rgb = proc.create_change_mask_visualization(
            results['confidence'],
            colormap='fire'  # Better for burn areas
        )

        # Create overlay with high opacity for better visibility
        overlay_rgb = proc.create_overlay_visualization(
            post_rgb,
            results['confidence'],
            alpha=0.85
        )

        # Create boxed visualization (red squares) - SEPARATE OUTPUT
        boxed_rgb = proc.create_boxed_visualization(
            post_rgb,
            results['confidence']
        )

        # Save results
        logger.info("Saving results...")
        output_paths = proc.save_results(
            pre_rgb=pre_rgb,
            post_rgb=post_rgb,
            mask_rgb=mask_rgb,
            mask_geotiff=results['mask'],
            output_dir=app.config['RESULTS_FOLDER'],
            profile=pre_data['profile'],
            prefix=timestamp
        )

        # Save overlay
        from PIL import Image
        overlay_path = os.path.join(
            app.config['RESULTS_FOLDER'],
            f'{timestamp}_overlay.png'
        )
        Image.fromarray(overlay_rgb).save(overlay_path)
        output_paths['overlay_png'] = overlay_path

        # Save boxed output
        boxed_path = os.path.join(
            app.config['RESULTS_FOLDER'],
            f'{timestamp}_boxed.png'
        )
        Image.fromarray(boxed_rgb).save(boxed_path)
        output_paths['boxed_png'] = boxed_path

        # Calculate statistics
        total_pixels = results['mask'].size
        burned_pixels = results['mask'].sum()
        burn_percentage = (burned_pixels / total_pixels) * 100
        
        # Get severity stats if available
        severity_stats = results.get('stats', {})

        logger.info(
            f"Burn detection complete! "
            f"Burned: {burned_pixels}/{total_pixels} "
            f"({burn_percentage:.2f}%)"
        )

        # Return results
        return jsonify({
            'success': True,
            'results': {
                'pre_image': f'/results/{os.path.basename(output_paths["pre_png"])}',
                'post_image': f'/results/{os.path.basename(output_paths["post_png"])}',
                'mask_image': f'/results/{os.path.basename(output_paths["mask_png"])}',
                'overlay_image': f'/results/{os.path.basename(overlay_path)}',
                'boxed_image': f'/results/{os.path.basename(boxed_path)}',
                'mask_geotiff': f'/results/{os.path.basename(output_paths["mask_geotiff"])}',
            },
            'statistics': {
                'total_pixels': int(total_pixels),
                'burned_pixels': int(burned_pixels),
                'burn_percentage': round(burn_percentage, 2),
                'threshold_used': threshold,
                'severity': severity_stats  # Burn severity breakdown
            }
        })

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({
            'error': f'Validation error: {str(e)}'
        }), 400

    except Exception as e:
        logger.error(f"Error during detection: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'Processing error: {str(e)}'
        }), 500


@app.route('/results/<path:filename>')
def serve_result(filename):
    """Serve result files."""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({
        'error': 'File too large. Maximum size is 500MB.'
    }), 413


def main():
    """Main entry point."""
    # Create necessary directories
    create_directories()

    # Initialize model at startup (optional, for faster first request)
    logger.info("Starting Prithivi Change Detection Server...")
    logger.info("Initializing model (this may take a few minutes)...")

    try:
        get_detector()
        logger.info("Model initialized successfully!")
    except Exception as e:
        logger.warning(f"Model initialization failed: {e}")
        logger.warning("Model will be loaded on first request.")

    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Set to True for development
        threaded=False  # Single-threaded for CPU inference
    )


if __name__ == '__main__':
    main()
