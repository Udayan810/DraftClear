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

// Event Listeners
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('drop', handleDrop);
uploadArea.addEventListener('click', () => fileInput.click());
fileButton.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
clearButton.addEventListener('click', clearFile);
processButton.addEventListener('click', processImage);
newProcessButton.addEventListener('click', resetUI);

// Drag and Drop
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.style.borderColor = '#E67E22';
    uploadArea.style.backgroundColor = 'rgba(230, 126, 34, 0.1)';
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

    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file (PNG, JPG, BMP)');
        return;
    }

    selectedFile = file;
    displayPreview(file);
    processButton.disabled = false;
}

function displayPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewSection.classList.remove('hidden');
        uploadArea.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function clearFile() {
    selectedFile = null;
    fileInput.value = '';
    previewSection.classList.add('hidden');
    uploadArea.style.display = '';
    processButton.disabled = true;
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

    try {
        // Create FormData
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('output_name', outputName);

        // Show processing steps animation
        animateProgressSteps();

        // Call API
        const response = await fetch(`${API_BASE_URL}/process`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Processing failed');
        }

        const result = await response.json();

        // Display results
        displayResults(result);
        showSuccess('Image processed successfully!');

    } catch (error) {
        showError(`Processing failed: ${error.message}`);
        progressSection.classList.add('hidden');
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

    // Update metrics
    document.getElementById('metricIterations').textContent = result.iterations;
    document.getElementById('metricLabels').textContent = result.text_labels;
    document.getElementById('metricCollisions').textContent = result.collision_count;
    document.getElementById('metricDecision').textContent = result.supervisor_decision === 'compile' ? '✓ Complete' : '⚠ In Progress';

    // Display images
    if (result.original_image) {
        document.getElementById('originalImage').src = `data:image/png;base64,${result.original_image}`;
    }
    if (result.healed_image) {
        document.getElementById('processedImage').src = `data:image/png;base64,${result.healed_image}`;
    }

    // Update download links
    if (result.pdf_url) {
        document.getElementById('downloadPdf').href = result.pdf_url;
        document.getElementById('downloadPdf').download = `${currentProcessName}.pdf`;
    }
    if (result.comparison_url) {
        document.getElementById('downloadComparison').href = result.comparison_url;
        document.getElementById('downloadComparison').download = `${currentProcessName}_comparison.png`;
    }

    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }, 500);
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
    checkApiHealth();
});

async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            console.log('API server is healthy');
        }
    } catch (error) {
        console.warn('API server not available:', error);
    }
}

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});
