const loadingIndicator = document.getElementById('loadingIndicator');
const generalInfoPanel = document.getElementById('generalInfoPanel');
const sceneEl = document.querySelector('a-scene');
const arCamera = document.getElementById('arCamera');
const catTextLabelContainer = document.getElementById('catTextLabelContainer_sceneLevel');

const PREDICTION_INTERVAL_MS = 10;
let predictionIntervalId = null;
let isProcessingFrame = false;
const MAX_LABELS_DISPLAY = 1;
let videoElement = null;
let arTextEntities = [];

let arTextData = [];
const SMOOTHING_FACTOR = 0.1;

function initAREntities() {
    console.log("Attempting to initAREntities...");
    if (!catTextLabelContainer_sceneLevel) {
        console.error("CRITICAL: Text label container (catTextLabelContainer_sceneLevel) NOT FOUND in HTML!");
        return;
    }
    arTextEntities = [];
    arTextData = [];
    for (let i = 0; i < MAX_LABELS_DISPLAY; i++) {
        const textEl = document.getElementById(`catText_${i}`);
        if (textEl) {
            arTextEntities.push(textEl);
            arTextData.push({
                entity: textEl,
                targetPosition: new THREE.Vector3(),
                currentPosition: new THREE.Vector3(),
                isVisible: false,
                targetValue: "",
                needsUpdate: false
            });
            const initialPos = textEl.getAttribute('position');
            if (initialPos) {
                arTextData[i].currentPosition.set(initialPos.x, initialPos.y, initialPos.z);
                arTextData[i].targetPosition.set(initialPos.x, initialPos.y, initialPos.z);
            }
            console.log(`Text entity catText_${i} found and added.`);
        } else {
            console.warn(`Text entity catText_${i} NOT FOUND in HTML.`);
        }
    }
    console.log(`Initialized ${arTextEntities.length} A-Frame text entities. Expected: ${MAX_LABELS_DISPLAY}`);
    if (arTextEntities.length === 0 && MAX_LABELS_DISPLAY > 0) {
        console.error("CRITICAL: No text entities were initialized. Check HTML IDs.");
    }
}

function hideAllAREntities() {
    arTextData.forEach(data => {
        if (data.entity) {
            data.entity.setAttribute('visible', 'false');
            data.isVisible = false;
            data.needsUpdate = false;
        }
    });
}

async function captureFrameForPredictionARJS() {
    if (!videoElement) {
        return null;
    }
    if (videoElement.readyState < videoElement.HAVE_ENOUGH_DATA || videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
        console.warn("captureFrame: Video not ready or dimensions are zero.",
            `ReadyState: ${videoElement.readyState}, Width: ${videoElement.videoWidth}, Height: ${videoElement.videoHeight}`);
        return null;
    }

    return new Promise((resolve) => {
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(blob => {
            if (blob) {
                resolve(blob);
            } else {
                console.error("captureFrame: canvas.toBlob failed to produce a blob.");
                resolve(null);
            }
        }, 'image/jpeg', 0.7);
    });
}

function screenToWorldCoordinates(screenX, screenY, distance = 1.5) {
    if (!arCamera) {
        console.error("screenToWorldCoordinates: arCamera is null.");
        return null;
    }
    if (!videoElement || videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
        console.warn("screenToWorldCoordinates: Video element not ready or dimensions are zero.",
            `Video Width: ${videoElement ? videoElement.videoWidth : 'N/A'}, Height: ${videoElement ? videoElement.videoHeight : 'N/A'}`);
        return null;
    }

    const threeCamera = arCamera.getObject3D('camera');
    if (!threeCamera) {
        console.error("screenToWorldCoordinates: THREE.js camera object not found in A-Frame camera.");
        return null;
    }

    threeCamera.updateMatrixWorld(true);

    const ndcX = (screenX / videoElement.videoWidth) * 2 - 1;
    const ndcY = -(screenY / videoElement.videoHeight) * 2 + 1;

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera({ x: ndcX, y: ndcY }, threeCamera);

    const targetPosition = new THREE.Vector3();
    raycaster.ray.at(distance, targetPosition);

    if (isNaN(targetPosition.x) || isNaN(targetPosition.y) || isNaN(targetPosition.z)) {
        console.error("screenToWorldCoordinates: Resulting world position contains NaN values. This is a critical error.", targetPosition);
        return null;
    }
    return targetPosition;
}


async function performRealtimePrediction() {
    if (isProcessingFrame) {
        return;
    }
    if (!arCamera) {
        console.warn("performRealtimePrediction: Skipping, arCamera not available.");
        return;
    }
    isProcessingFrame = true;
    if (loadingIndicator) loadingIndicator.style.display = 'block';

    const imageBlob = await captureFrameForPredictionARJS();

    if (!imageBlob) {
        if (loadingIndicator) loadingIndicator.style.display = 'none';
        isProcessingFrame = false;
        return;
    }

    const formData = new FormData();
    formData.append('image', imageBlob, 'capture.jpg');

    try {
        const response = await fetch('/predict_ar_realtime', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        console.log("performRealtimePrediction: Server response:", JSON.stringify(data, null, 2));

        arTextData.forEach(labelData => labelData.needsUpdate = false);

        hideAllAREntities();
        if (generalInfoPanel) generalInfoPanel.textContent = "Arahkan kamera ke kucing...";

        if (data.error) {
            console.error('performRealtimePrediction: Server Error:', data.error);
            if (generalInfoPanel) generalInfoPanel.textContent = 'Error: ' + data.error;
        } else if (data.predictions && data.predictions.length > 0) {
            console.log(`performRealtimePrediction: Received ${data.predictions.length} predictions.`);
            const firstCat = data.predictions[0];
            if (generalInfoPanel) {
                generalInfoPanel.innerHTML = `<strong>${firstCat.breed}</strong> (${(firstCat.probability * 100).toFixed(1)}%)<br><small>${firstCat.description}</small>`;
            }

            data.predictions.slice(0, MAX_LABELS_DISPLAY).forEach((catPred, index) => {
                const labelData = arTextData[index];
                if (!labelData || !labelData.entity) {
                    console.warn(`performRealtimePrediction: Text entity data at index ${index} is undefined. Skipping.`);
                    return;
                }
                console.log(`performRealtimePrediction: Processing prediction ${index} for breed ${catPred.breed}`);

                const [x1, y1, x2, y2] = catPred.bbox_2d;
                const bboxCenterX = (x1 + x2) / 2;
                const bboxTopY = y1 - 10;

                console.log(`performRealtimePrediction: Cat ${index} - BBox CenterX: ${bboxCenterX.toFixed(0)}, TopY: ${bboxTopY.toFixed(0)}`);

                const worldPos = screenToWorldCoordinates(bboxCenterX, bboxTopY, 1.8);

                if (worldPos) {
                    labelData.targetPosition.copy(worldPos);
                    const breedText = `${catPred.breed} (${(catPred.probability * 100).toFixed(0)}%)`;
                    const shortDescription = catPred.description.length > 200 ? catPred.description.substring(0, 197) + "..." : catPred.description;
                    labelData.targetValue = `${breedText}\n${shortDescription}`;
                    labelData.isVisible = true;
                    labelData.needsUpdate = true;
                    console.log(`performRealtimePrediction: Text entity ${index} ('${breedText}') targetPos set to [${worldPos.x.toFixed(2)}, ${worldPos.y.toFixed(2)}, ${worldPos.z.toFixed(2)}]`);
                } else {
                    labelData.isVisible = false;
                    console.warn(`performRealtimePrediction: Cat ${index} - worldPos is null. Text will be hidden.`);
                }
            });
        } else {
             if (generalInfoPanel) generalInfoPanel.textContent = "Tidak ada kucing terdeteksi.";
             console.log("performRealtimePrediction: No cat predictions in server response or predictions array is empty.");
        }

        arTextData.forEach(labelData => {
            if (!labelData.needsUpdate && labelData.isVisible) {
                labelData.isVisible = false;
            }
        });
    } catch (error) {
        console.error('performRealtimePrediction: Fetch Error or other JS error in try block:', error);
        if (generalInfoPanel) generalInfoPanel.textContent = 'Gagal menghubungi server atau error JS.';
    } finally {
        if (loadingIndicator) loadingIndicator.style.display = 'none';
        isProcessingFrame = false;
    }
}

function updateTextEntities() {
    if (!arCamera || arTextData.length === 0) return;

    const cameraWorldPosition = new THREE.Vector3();
    arCamera.object3D.getWorldPosition(cameraWorldPosition);

    arTextData.forEach(labelData => {
        if (!labelData.entity) return;

        if (labelData.isVisible) {
            labelData.currentPosition.lerp(labelData.targetPosition, SMOOTHING_FACTOR);

            labelData.entity.setAttribute('position', {
                x: labelData.currentPosition.x,
                y: labelData.currentPosition.y,
                z: labelData.currentPosition.z
            });

            if (labelData.entity.getAttribute('value') !== labelData.targetValue) {
                labelData.entity.setAttribute('value', labelData.targetValue);
            }

            labelData.entity.setAttribute('look-at', cameraWorldPosition);
            if (labelData.entity.getAttribute('visible') === false || labelData.entity.getAttribute('visible') === 'false') {
                 labelData.entity.setAttribute('visible', 'true');
            }
        } else {
            if (labelData.entity.getAttribute('visible') === true || labelData.entity.getAttribute('visible') === 'true') {
                 labelData.entity.setAttribute('visible', 'false');
            }
        }
    });

    requestAnimationFrame(updateTextEntities);
}

sceneEl.addEventListener('loaded', function () {
    console.log("A-Frame scene 'loaded' event fired.");
    initAREntities();

    setTimeout(() => {
        console.log("setTimeout after scene 'loaded' for video search.");
        const videos = document.getElementsByTagName('video');
        let arVideo = null;
        if (videos.length > 0) {
            console.log(`Found ${videos.length} video element(s).`);
            for (let i = 0; i < videos.length; i++) {
                const v = videos[i];
                console.log(`Checking video ${i}: srcObject active=${v.srcObject ? v.srcObject.active : 'N/A'}, readyState=${v.readyState}`);
                if (v.srcObject && v.srcObject.active && v.offsetWidth > 0 && v.offsetHeight > 0) {
                    arVideo = v;
                    console.log(`AR.js Video element candidate found (video ${i}).`);
                    break;
                }
            }
             if(!arVideo && videos.length > 0) arVideo = videos[0];
        } else {
            console.error("CRITICAL: No video elements found on the page.");
            return;
        }

        if (arVideo) {
            videoElement = arVideo;
            console.log("AR.js Video element assigned:", videoElement);
            console.log(`Video initial dimensions: Width=${videoElement.videoWidth}, Height=${videoElement.videoHeight}, ReadyState=${videoElement.readyState}`);

            const startPredictionLoop = () => {
                console.log(`Attempting to start prediction loop. Video dimensions: W=${videoElement.videoWidth}, H=${videoElement.videoHeight}, ReadyState=${videoElement.readyState}`);
                if (videoElement.videoWidth > 0 && videoElement.videoHeight > 0 && videoElement.readyState >= videoElement.HAVE_ENOUGH_DATA) {
                    console.log("SUCCESS: Video AR.js ready with dimensions. Starting prediction loop.");
                    if (predictionIntervalId) clearInterval(predictionIntervalId);
                    predictionIntervalId = setInterval(performRealtimePrediction, PREDICTION_INTERVAL_MS);
                } else {
                    console.warn("Video AR.js not fully ready or dimensions are zero. Will retry check or wait for event.",
                        `W:${videoElement.videoWidth}, H:${videoElement.videoHeight}, State:${videoElement.readyState}`);
                }
            };

            if (videoElement.readyState >= videoElement.HAVE_ENOUGH_DATA && videoElement.videoWidth > 0 && videoElement.videoHeight > 0) {
                 startPredictionLoop();
            } else {
                 console.log("Video not immediately ready, attaching event listeners: 'loadeddata', 'canplay', 'playing'");
                 videoElement.addEventListener('loadeddata', startPredictionLoop);
                 videoElement.addEventListener('canplay', startPredictionLoop);
                 videoElement.addEventListener('playing', startPredictionLoop);
                 setTimeout(startPredictionLoop, 3000);
            }
        } else {
            console.error("CRITICAL: Failed to identify a suitable AR.js video element.");
        }
    }, 2500);

    requestAnimationFrame(updateTextEntities);
    console.log("Started text entities update loop (requestAnimationFrame).");
});

window.addEventListener('beforeunload', () => {
    if (predictionIntervalId) {
        clearInterval(predictionIntervalId);
        predictionIntervalId = null;
        console.log("Prediction interval cleared on page unload.");
    }
});

console.log("arjs-logic.js loaded.");