/**
 * Frontend JavaScript for Wildfire Damage Detection
 */

// DOM Elements
const uploadForm = document.getElementById('uploadForm');
const submitBtn = document.getElementById('submitBtn');
const preImageInput = document.getElementById('preImage');
const postImageInput = document.getElementById('postImage');
const thresholdInput = document.getElementById('threshold');
const thresholdValue = document.getElementById('thresholdValue');

// Section elements
const progressSection = document.getElementById('progressSection');
const errorSection = document.getElementById('errorSection');
const resultsSection = document.getElementById('resultsSection');

// Info elements
const preFileInfo = document.getElementById('preFileInfo');
const postFileInfo = document.getElementById('postFileInfo');

// Result elements
const preResultImage = document.getElementById('preResultImage');
const postResultImage = document.getElementById('postResultImage');
const maskResultImage = document.getElementById('maskResultImage');
const overlayResultImage = document.getElementById('overlayResultImage');
const boxedResultImage = document.getElementById('boxedResultImage');

// Statistics elements
const burnedPixelsEl = document.getElementById('changedPixels'); // Mapping to existing ID for now
const totalPixelsEl = document.getElementById('totalPixels');
const burnPercentageEl = document.getElementById('changePercentage'); // Mapping to existing ID for now
const thresholdUsedEl = document.getElementById('thresholdUsed');

// Severity section (to be added)
let severityContainer;

// Download links
const downloadPre = document.getElementById('downloadPre');
const downloadPost = document.getElementById('downloadPost');
const downloadMask = document.getElementById('downloadMask');
const downloadOverlay = document.getElementById('downloadOverlay');
const downloadBoxed = document.getElementById('downloadBoxed');
const downloadGeoTIFF = document.getElementById('downloadGeoTIFF');

/**
 * Initialize event listeners
 */
function init() {
    // Form submission
    uploadForm.addEventListener('submit', handleSubmit);

    // File input changes
    preImageInput.addEventListener('change', () => handleFileSelect(preImageInput, preFileInfo));
    postImageInput.addEventListener('change', () => handleFileSelect(postImageInput, postFileInfo));

    // Threshold slider
    thresholdInput.addEventListener('input', updateThresholdDisplay);
}

/**
 * Handle file selection and display info
 */
function handleFileSelect(input, infoElement) {
    const file = input.files[0];
    if (file) {
        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
        infoElement.textContent = `${file.name} (${sizeMB} MB)`;
        infoElement.style.color = '#10b981';
    } else {
        infoElement.textContent = '';
    }
}

/**
 * Update threshold display value
 */
function updateThresholdDisplay() {
    thresholdValue.textContent = thresholdInput.value;
}

/**
 * Handle form submission
 */
async function handleSubmit(e) {
    e.preventDefault();

    // Validate files
    if (!preImageInput.files[0] || !postImageInput.files[0]) {
        showError('Please select both pre and post event images.');
        return;
    }

    // Validate file extensions
    const preFile = preImageInput.files[0];
    const postFile = postImageInput.files[0];

    if (!isValidGeoTIFF(preFile.name) || !isValidGeoTIFF(postFile.name)) {
        showError('Please upload valid GeoTIFF files (.tif or .tiff)');
        return;
    }

    // Prepare form data
    const formData = new FormData();
    formData.append('pre_image', preFile);
    formData.append('post_image', postFile);
    formData.append('threshold', thresholdInput.value);

    // Show progress
    showProgress();

    try {
        // Submit to server
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.success) {
            // Show results
            displayResults(data);
        } else {
            // Show error
            showError(data.error || 'An error occurred during processing.');
        }
    } catch (error) {
        console.error('Error:', error);
        if (error instanceof TypeError && error.message.includes('toString')) {
            showError('Error displaying results. The processing finished but the frontend failed to parse the results.');
        } else {
            showError('Network error or processing timeout. If this persists, please check the uploads/results folder.');
        }
    }
}

/**
 * Validate GeoTIFF file extension
 */
function isValidGeoTIFF(filename) {
    const ext = filename.toLowerCase().split('.').pop();
    return ext === 'tif' || ext === 'tiff';
}

/**
 * Show progress section
 */
function showProgress() {
    // Hide all sections
    errorSection.style.display = 'none';
    resultsSection.style.display = 'none';

    // Show progress
    progressSection.style.display = 'block';

    // Disable submit button
    submitBtn.disabled = true;
    submitBtn.textContent = 'Processing...';

    // Scroll to progress
    progressSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

/**
 * Show error section
 */
function showError(message) {
    // Hide other sections
    progressSection.style.display = 'none';
    resultsSection.style.display = 'none';

    // Show error
    errorSection.style.display = 'block';
    document.getElementById('errorMessage').textContent = message;

    // Re-enable submit button
    submitBtn.disabled = false;
    submitBtn.textContent = 'Detect Changes';

    // Scroll to error
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

/**
 * Display detection results
 */
function displayResults(data) {
    // Hide other sections
    progressSection.style.display = 'none';
    errorSection.style.display = 'none';

    // Update statistics using new burn detection keys
    burnedPixelsEl.textContent = formatNumber(data.statistics.burned_pixels || 0);
    totalPixelsEl.textContent = formatNumber(data.statistics.total_pixels || 0);
    burnPercentageEl.textContent = `${data.statistics.burn_percentage || 0}%`;
    thresholdUsedEl.textContent = data.statistics.threshold_used;

    // Display severity breakdown if available
    if (data.statistics.severity) {
        updateSeverityStats(data.statistics.severity);
    }

    // Update images
    preResultImage.src = data.results.pre_image;
    postResultImage.src = data.results.post_image;
    maskResultImage.src = data.results.mask_image;
    overlayResultImage.src = data.results.overlay_image;
    boxedResultImage.src = data.results.boxed_image;

    // Update download links
    downloadPre.href = data.results.pre_image;
    downloadPost.href = data.results.post_image;
    downloadMask.href = data.results.mask_image;
    downloadOverlay.href = data.results.overlay_image;
    downloadBoxed.href = data.results.boxed_image;
    downloadGeoTIFF.href = data.results.mask_geotiff;

    // Show results
    resultsSection.style.display = 'block';

    // Re-enable submit button
    submitBtn.disabled = false;
    submitBtn.textContent = 'Detect Changes';

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Format number with commas
 */
function formatNumber(num) {
    if (num === undefined || num === null) return '0';
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

/**
 * Update severity statistics visualization
 */
function updateSeverityStats(severity) {
    // Look for or create severity box
    let severityDiv = document.getElementById('severityStats');
    if (!severityDiv) {
        severityDiv = document.createElement('div');
        severityDiv.id = 'severityStats';
        severityDiv.className = 'severity-stats';
        document.querySelector('.statistics').after(severityDiv);
    }

    severityDiv.innerHTML = `
        <h3>Burn Severity Breakdown</h3>
        <div class="severity-bars">
            ${createSeverityBar('Unburned', severity.unburned_pct, '#10b981')}
            ${createSeverityBar('Low', severity.low_severity_pct, '#fbbf24')}
            ${createSeverityBar('Moderate-Low', severity.moderate_low_pct, '#f59e0b')}
            ${createSeverityBar('Moderate-High', severity.moderate_high_pct, '#ef4444')}
            ${createSeverityBar('High', severity.high_severity_pct, '#b91c1c')}
        </div>
    `;
}

function createSeverityBar(label, pct, color) {
    return `
        <div class="severity-item">
            <div class="severity-label">
                <span>${label}</span>
                <span>${pct.toFixed(2)}%</span>
            </div>
            <div class="progress-bar-bg" style="background: rgba(255,255,255,0.1); border-radius: 4px; height: 8px; margin-top: 4px;">
                <div class="progress-bar-fill" style="width: ${pct}%; background: ${color}; height: 100%; border-radius: 4px;"></div>
            </div>
        </div>
    `;
}

/**
 * Reset form to initial state
 */
function resetForm() {
    // Reset form
    uploadForm.reset();
    preFileInfo.textContent = '';
    postFileInfo.textContent = '';
    thresholdValue.textContent = '0.5';

    // Hide all sections
    progressSection.style.display = 'none';
    errorSection.style.display = 'none';
    resultsSection.style.display = 'none';

    // Re-enable submit button
    submitBtn.disabled = false;
    submitBtn.textContent = 'Detect Changes';

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/**
 * Check server health on load
 */
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();

        if (response.ok) {
            console.log('Server health:', data);
        } else {
            console.warn('Server health check failed:', data);
        }
    } catch (error) {
        console.error('Health check error:', error);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    init();
    checkHealth();
});

// Image zoom functionality (bonus feature)
document.addEventListener('click', (e) => {
    if (e.target.tagName === 'IMG' && e.target.closest('.image-container')) {
        const img = e.target;

        // Create modal for full-size view
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 9999;
            cursor: zoom-out;
        `;

        const modalImg = document.createElement('img');
        modalImg.src = img.src;
        modalImg.style.cssText = `
            max-width: 95%;
            max-height: 95%;
            object-fit: contain;
        `;

        modal.appendChild(modalImg);
        document.body.appendChild(modal);

        // Close on click
        modal.addEventListener('click', () => {
            document.body.removeChild(modal);
        });
    }
});
