/* DraftClear Frontend Script */

const API_BASE_URL = '/api';
let selectedFile = null;
let currentProcessName = '';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileButton = document.getElementById('fileButton');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const clearButton = document.getElementById('clearButton');
const outputNameInput = document.getElementById('outputName');
const processButton = document.getElementById('processButton');
const progressSection = document.getElementById('progressSection');
const resultsSection = document.getElementById('resultsSection');
const newProcessButton = document.getElementById('newProcessButton');
const originalResultImage = document.getElementById('originalImage');
const processedResultImage = document.getElementById('processedImage');
const downloadPdfButton = document.getElementById('downloadPdf');
const downloadComparisonButton = document.getElementById('downloadComparison');
const logoutButton = document.getElementById('logoutButton');


// Event Listeners
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('drop', handleDrop);
uploadArea.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
clearButton.addEventListener('click', clearFile);
processButton.addEventListener('click', processImage);
newProcessButton.addEventListener('click', resetUI);
downloadPdfButton.addEventListener('click', handleDownloadClick);
downloadComparisonButton.addEventListener('click', handleDownloadClick);
if (logoutButton) {
    logoutButton.addEventListener('click', (e) => {
        e.preventDefault();
        localStorage.removeItem('dc_token');
        window.location.href = 'login.html';
    });
}

// Drag and Drop
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.style.borderColor = '#00338D';
    uploadArea.style.backgroundColor = 'rgba(0, 51, 141, 0.1)';
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.style.borderColor = '';
    uploadArea.style.backgroundColor = '';

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        handleFileSelect();
    }
}

// File Selection
function handleFileSelect() {
    const file = fileInput.files[0];
    if (!file) return;

    // Validate file type - support images and CAD formats
    const validImageTypes = ['image/png', 'image/jpeg', 'image/bmp', 'image/gif', 'image/webp'];
    const validCADExtensions = ['.dwg', '.dxf', '.DWG', '.DXF'];
    const fileName = file.name.toLowerCase();

    const isValidImage = validImageTypes.includes(file.type);
    const isValidCAD = validCADExtensions.some(ext => fileName.endsWith(ext));

    if (!isValidImage && !isValidCAD) {
        showError('Please select a valid file: Image (PNG, JPG, BMP) or CAD (DWG, DXF)');
        return;
    }

    selectedFile = file;
    displayPreview(file);
    processButton.disabled = false;
}

function displayPreview(file) {
    const isCAD = ['.dwg', '.dxf'].some(ext => file.name.toLowerCase().endsWith(ext));

    if (isCAD) {
        // CAD files can't be rendered as images in the browser — show a placeholder
        previewImage.style.display = 'none';

        // Remove any existing CAD placeholder
        const existing = document.getElementById('cadPlaceholder');
        if (existing) existing.remove();

        const isDWG = file.name.toLowerCase().endsWith('.dwg');

        const placeholder = document.createElement('div');
        placeholder.id = 'cadPlaceholder';

        if (isDWG) {
            // DWG: blue caution card — file IS accepted
            placeholder.style.cssText = `
                display: flex; flex-direction: column; align-items: center;
                justify-content: center; gap: 10px; padding: 28px 20px;
                background: rgba(0,51,141,0.06); border: 2px dashed #00338D;
                border-radius: 12px; margin: 12px 0; color: #00338D;
            `;
            placeholder.innerHTML = `
                <i class="fas fa-drafting-compass" style="font-size: 2.6rem; color:#00338D;"></i>
                <div style="font-weight: 700; font-size: 1rem; color:#00338D;">DWG File Loaded</div>
                <div style="font-size: 0.82rem; color: #555; text-align:center; line-height:1.7;">
                    <strong>${file.name}</strong> &nbsp;·&nbsp; ${(file.size / 1024).toFixed(1)} KB &nbsp;·&nbsp; DWG format<br>
                    <span style="color:#00338D; font-weight:600;">✓ DWG File supported — clearing conflicts internally</span><br>
                    <span style="font-size:0.78rem; color:#888;">
                        Direct processing enabled via internal converter.
                    </span>
                </div>
            `;
        } else {
            // DXF: fully supported
            placeholder.style.cssText = `
                display: flex; flex-direction: column; align-items: center;
                justify-content: center; gap: 12px; padding: 40px 20px;
                background: rgba(39,174,96,0.07); border: 2px dashed #27ae60;
                border-radius: 12px; margin: 12px 0; color: #27ae60;
            `;
            placeholder.innerHTML = `
                <i class="fas fa-drafting-compass" style="font-size: 3rem;"></i>
                <div style="font-weight: 600; font-size: 1rem;">${file.name}</div>
                <div style="font-size: 0.85rem; color: #888; text-align:center;">
                    DXF file ready for processing<br>
                    <span style="font-size:0.8rem;">${(file.size / 1024).toFixed(1)} KB &nbsp;·&nbsp; DXF format</span>
                </div>
            `;
        }

        // Insert placeholder inside the preview section
        const previewContainer = previewSection.querySelector('.preview-header');
        previewContainer.insertAdjacentElement('afterend', placeholder);
    } else {
        // It's an image — render it normally
        const existing = document.getElementById('cadPlaceholder');
        if (existing) existing.remove();
        previewImage.style.display = '';

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    previewSection.classList.remove('hidden');
    uploadArea.style.display = 'none';
}

function clearFile() {
    selectedFile = null;
    fileInput.value = '';

    // Remove CAD placeholder if present
    const cadPlaceholder = document.getElementById('cadPlaceholder');
    if (cadPlaceholder) cadPlaceholder.remove();
    previewImage.style.display = '';
    previewImage.src = '';

    previewSection.classList.add('hidden');
    uploadArea.style.display = '';
    processButton.disabled = true;
    originalResultImage.src = '';
    processedResultImage.src = '';
    setDownloadTarget(downloadPdfButton, '', '');
    setDownloadTarget(downloadComparisonButton, '', '');
}

// Process Image
async function processImage() {
    if (!selectedFile) {
        showError('Please select an image file');
        return;
    }

    const outputName = outputNameInput.value.trim() || `drawing_${Date.now()}`;
    currentProcessName = outputName;

    // Show progress section
    progressSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');

    // Button loading state
    const originalBtnContent = processButton.innerHTML;
    processButton.disabled = true;
    processButton.classList.add('btn-loading');
    processButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';

    try {
        // Create FormData
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('output_name', outputName);

        // Show processing steps animation
        animateProgressSteps();

        // Call API with Auth Header
        const token = localStorage.getItem('dc_token');
        const headers = {};
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        const response = await fetch(`${API_BASE_URL}/process`, {
            method: 'POST',
            headers: headers,
            body: formData
        });

        if (response.status === 401 || response.status === 403) {
            localStorage.removeItem('dc_token');
            window.location.href = 'login.html';
            return;
        }

        if (!response.ok) {
            const error = await response.json();
            // Format multi-line server error messages (e.g. DWG instructions)
            const detail = error.detail || 'Processing failed';
            throw new Error(detail);
        }

        const result = await response.json();

        // Display results
        displayResults(result);
        showSuccess('Image processed successfully!');

    } catch (error) {
        showError(`Processing failed: ${error.message}`);
        progressSection.classList.add('hidden');
    } finally {
        // Restore button state
        processButton.disabled = false;
        processButton.classList.remove('btn-loading');
        processButton.innerHTML = originalBtnContent;
    }
}

function animateProgressSteps() {
    const steps = [1, 2, 3, 4, 5];
    let currentStep = 0;

    const interval = setInterval(() => {
        if (currentStep > 0) {
            document.getElementById(`step${currentStep}`).classList.remove('active');
        }
        if (currentStep < steps.length) {
            document.getElementById(`step${currentStep + 1}`).classList.add('active');
            currentStep++;
        } else {
            clearInterval(interval);
        }
    }, 800);
}

function displayResults(result) {
    progressSection.classList.add('hidden');
    resultsSection.classList.remove('hidden');

    // For simplicity, we display metrics for the first page in the main view
    const mainPage = result.pages[0] || {};
    
    // Update metrics
    document.getElementById('metricIterations').textContent = mainPage.iterations || 0;
    document.getElementById('metricLabels').textContent = mainPage.text_labels || 0;
    document.getElementById('metricCollisions').textContent = mainPage.collision_count || 0;
    document.getElementById('metricDecision').textContent = 
        result.total_pages > 1 ? `✓ ${result.total_pages} Pages` : (mainPage.supervisor_decision === 'compile' ? '✓ Complete' : '⚠ In Progress');

    // Display images (Page 1 previews)
    const cacheBuster = `ts=${Date.now()}`;
    const originalImageUrl = mainPage.original_image_url ? `${mainPage.original_image_url}?${cacheBuster}` : '';
    const processedImageUrl = mainPage.processed_image_url ? `${mainPage.processed_image_url}?${cacheBuster}` : '';

    setResultImage(originalResultImage, originalImageUrl, '');
    setResultImage(processedResultImage, processedImageUrl, originalImageUrl);

    // Update download links
    setDownloadTarget(downloadPdfButton, '', '');
    setDownloadTarget(downloadComparisonButton, '', '');

    if (result.pdf_url) {
        setDownloadTarget(downloadPdfButton, result.pdf_url, `${currentProcessName}.pdf`);
    }
    if (result.comparison_url) {
        setDownloadTarget(downloadComparisonButton, result.comparison_url, `${currentProcessName}_comparison.png`);
    }

    // Show warning if 0 labels detected (common with watermarked evaluation files)
    if (result.text_labels === 0) {
        showSuccessNotification("Processing complete, but 0 labels were detected. This may happen if the file is watermarked or uses an unsupported version.");
    }

    // Scroll with a slight delay to ensure images have started loading
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 500);
}

function setResultImage(imageElement, primarySource, fallbackSource) {
    if (!imageElement) return;
    
    // Reset state and clear previous image while loading
    imageElement.onerror = null;
    imageElement.style.opacity = '0'; // Hide while loading
    
    if (!primarySource && !fallbackSource) {
        imageElement.src = '';
        return;
    }

    // When image loads successfully
    imageElement.onload = () => {
        imageElement.style.opacity = '1';
        imageElement.style.transition = 'opacity 0.4s ease';
    };

    // If primary fails, try fallback
    imageElement.onerror = () => {
        console.warn("Primary image load failed, trying fallback");
        imageElement.onerror = null;
        if (fallbackSource && primarySource !== fallbackSource) {
            imageElement.src = fallbackSource;
        } else {
            imageElement.style.opacity = '0.3'; // Dimmed state for failed image
        }
    };

    imageElement.src = primarySource || fallbackSource;
}

function setDownloadTarget(button, url, fallbackFilename) {
    button.dataset.downloadUrl = url || '';
    button.dataset.filename = fallbackFilename || '';
    button.href = url || '#';

    if (url) {
        button.classList.remove('disabled');
        button.setAttribute('aria-disabled', 'false');
    } else {
        button.classList.add('disabled');
        button.setAttribute('aria-disabled', 'true');
    }
}

function extractFilename(contentDisposition, fallbackFilename) {
    if (!contentDisposition) {
        return fallbackFilename;
    }

    const utf8Match = contentDisposition.match(/filename\*=UTF-8''([^;]+)/i);
    if (utf8Match) {
        return decodeURIComponent(utf8Match[1]);
    }

    const basicMatch = contentDisposition.match(/filename="?([^";]+)"?/i);
    return basicMatch ? basicMatch[1] : fallbackFilename;
}

async function downloadFile(url, fallbackFilename) {
    const response = await fetch(url);
    if (!response.ok) {
        let errorMessage = 'Download failed';
        try {
            const error = await response.json();
            errorMessage = error.detail || errorMessage;
        } catch {
            // Ignore JSON parse errors and keep the generic message.
        }
        throw new Error(errorMessage);
    }

    const blob = await response.blob();
    const objectUrl = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = objectUrl;
    link.download = extractFilename(response.headers.get('content-disposition'), fallbackFilename);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => window.URL.revokeObjectURL(objectUrl), 1000);
}

async function handleDownloadClick(event) {
    event.preventDefault();

    const button = event.currentTarget;
    const url = button.dataset.downloadUrl;
    if (!url) {
        showError('This file is not ready to download yet.');
        return;
    }

    try {
        await downloadFile(url, button.dataset.filename);
    } catch (error) {
        showError(`Download failed: ${error.message}`);
    }
}

// Notifications
function showError(message) {
    const notification = document.getElementById('errorNotification');
    document.getElementById('errorMessage').textContent = message;
    notification.classList.remove('hidden');
    setTimeout(() => closeNotification(), 5000);
}

function showSuccess(message) {
    const notification = document.getElementById('successNotification');
    document.getElementById('successMessage').textContent = message;
    notification.classList.remove('hidden');
    setTimeout(() => closeNotification(), 5000);
}

function closeNotification() {
    document.getElementById('errorNotification').classList.add('hidden');
    document.getElementById('successNotification').classList.add('hidden');
}

// Reset UI
function resetUI() {
    clearFile();
    outputNameInput.value = '';
    progressSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    previewSection.classList.add('hidden');
    uploadArea.style.display = '';
}

// Health check on load
document.addEventListener('DOMContentLoaded', () => {
    if (!localStorage.getItem('dc_token')) {
        window.location.href = 'login.html';
        return;
    }
    setDownloadTarget(downloadPdfButton, '', '');
    setDownloadTarget(downloadComparisonButton, '', '');
    checkApiHealth();
});

async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            console.log('API is healthy');
        }
    } catch (error) {
        console.error('API health check failed:', error);
    }
}




// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {

        const href = this.getAttribute('href');

        if (!href || href === '#') {
            e.preventDefault();
            return;
        }

        if (!href.startsWith('#')) {
            return;
        }

        e.preventDefault();
        const target = document.querySelector(href);
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});
