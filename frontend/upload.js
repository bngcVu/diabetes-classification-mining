const API_BASE_URL = "http://localhost:5000";

// Mode switching
let currentMode = "predict";
let selectedPredictFile = null;
let selectedRetrainFile = null;
let retrainPollInterval = null;

// DOM elements
const modeTabs = document.querySelectorAll(".mode-tab");
const predictForm = document.getElementById("predictForm");
const retrainForm = document.getElementById("retrainForm");
const resultPanel = document.getElementById("resultPanel");
const predictResults = document.getElementById("predictResults");
const retrainResults = document.getElementById("retrainResults");
const formError = document.getElementById("formError");
const formSuccess = document.getElementById("formSuccess");

// Switch between modes
modeTabs.forEach(tab => {
  tab.addEventListener("click", () => {
    const mode = tab.dataset.mode;
    switchMode(mode);
  });
});

function switchMode(mode) {
  currentMode = mode;
  modeTabs.forEach(t => t.classList.toggle("active", t.dataset.mode === mode));
  predictForm.classList.toggle("active", mode === "predict");
  retrainForm.classList.toggle("active", mode === "retrain");
  
  // Hide results when switching
  resultPanel.hidden = true;
  formError.hidden = true;
  formSuccess.hidden = true;
}

// Setup drag and drop for predict mode
setupDropZone("predictDropZone", "predictFileInput", "predictFileInfo", "predictFileName", "predictFileRows", "predictSubmitBtn", 
  (file) => { selectedPredictFile = file; },
  () => { return selectedPredictFile; }
);

// Setup drag and drop for retrain mode
setupDropZone("retrainDropZone", "retrainFileInput", "retrainFileInfo", "retrainFileName", "retrainFileRows", "retrainSubmitBtn",
  (file) => { selectedRetrainFile = file; },
  () => { return selectedRetrainFile; }
);

function setupDropZone(zoneId, inputId, infoId, nameId, rowsId, btnId, onFileSelect, getFile) {
  const dropZone = document.getElementById(zoneId);
  const fileInput = document.getElementById(inputId);
  const fileInfo = document.getElementById(infoId);
  const submitBtn = document.getElementById(btnId);

  if (!dropZone || !fileInput) return;

  fileInput.addEventListener("change", (e) => {
    handleFile(e.target.files[0], fileInfo, nameId, rowsId, submitBtn, onFileSelect);
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    handleFile(e.dataTransfer.files[0], fileInfo, nameId, rowsId, submitBtn, onFileSelect);
  });

  dropZone.addEventListener("click", () => {
    fileInput.click();
  });
}

function handleFile(file, fileInfo, nameId, rowsId, submitBtn, onFileSelect) {
  formError.hidden = true;
  formSuccess.hidden = true;

  if (!file) return;

  if (!file.name.endsWith(".csv")) {
    showError("Chi chap nhan file CSV");
    return;
  }

  onFileSelect(file);
  document.getElementById(nameId).textContent = file.name;
  submitBtn.disabled = false;
  fileInfo.hidden = false;

  // Preview file to count rows
  const reader = new FileReader();
  reader.onload = (e) => {
    const lines = e.target.result.split("\n");
    const rowCount = Math.max(0, lines.length - 1);
    document.getElementById(rowsId).textContent = `${rowCount} dong`;
  };
  reader.readAsText(file);
}

function showError(msg) {
  formError.textContent = msg;
  formError.hidden = false;
}

function showSuccess(msg) {
  formSuccess.textContent = msg;
  formSuccess.hidden = false;
}

// Predict form submit
predictForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  if (!selectedPredictFile) {
    showError("Vui long chon file CSV");
    return;
  }

  const submitBtn = document.getElementById("predictSubmitBtn");
  submitBtn.disabled = true;
  submitBtn.textContent = "Dang xu ly...";
  formError.hidden = true;
  formSuccess.hidden = true;

  const formData = new FormData();
  formData.append("file", selectedPredictFile);
  formData.append("model", document.getElementById("predictModelSelect").value);

  try {
    const response = await fetch(`${API_BASE_URL}/predict-csv`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || data.details || "Upload that bai");
    }

    // Show results
    predictResults.hidden = false;
    retrainResults.hidden = true;
    resultPanel.hidden = false;

    document.getElementById("totalRecords").textContent = data.summary.total_records;
    document.getElementById("highRiskCount").textContent = data.summary.high_risk;
    document.getElementById("lowRiskCount").textContent = data.summary.low_risk;
    document.getElementById("highRiskPercent").textContent = data.summary.high_risk_percent + "%";

    // Setup download link
    const downloadLink = document.getElementById("downloadLink");
    downloadLink.href = `${API_BASE_URL}${data.download_url}`;
    downloadLink.download = data.download_path.split(/[\\/]/).pop();

    showSuccess(`Da xu ly ${data.summary.total_records} ban ghi thanh cong!`);

  } catch (error) {
    showError(error.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Du doan";
  }
});

// Retrain form submit
retrainForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  if (!selectedRetrainFile) {
    showError("Vui long chon file CSV");
    return;
  }

  const submitBtn = document.getElementById("retrainSubmitBtn");
  submitBtn.disabled = true;
  submitBtn.textContent = "Dang upload va kiem tra...";
  formError.hidden = true;
  formSuccess.hidden = true;

  // Show retrain results panel
  predictResults.hidden = true;
  retrainResults.hidden = false;
  resultPanel.hidden = false;

  // Reset progress steps
  resetProgressSteps();
  document.getElementById("retrainSummary").hidden = true;

  const formData = new FormData();
  formData.append("file", selectedRetrainFile);

  try {
    // Upload file
    updateProgressStep("stepValidate", "active");
    
    const response = await fetch(`${API_BASE_URL}/upload-train`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || data.details || "Upload that bai");
    }

    // Show initial success
    showSuccess(`Da upload ${data.validation.row_count} dong. Bat dau qua trinh huan luyen...`);
    
    // Start polling for status
    startRetrainPolling(data.validation);

  } catch (error) {
    showError(error.message);
    submitBtn.disabled = false;
    submitBtn.textContent = "🚀 Upload va Huan luyen lai Model";
  }
});

function resetProgressSteps() {
  const steps = ["stepValidate", "stepClean", "stepMerge", "stepTrain", "stepCompare"];
  steps.forEach(stepId => {
    const el = document.getElementById(stepId);
    if (el) {
      el.classList.remove("completed", "active");
    }
  });
}

function updateProgressStep(stepId, status) {
  const el = document.getElementById(stepId);
  if (el) {
    el.classList.remove("completed", "active");
    el.classList.add(status);
  }
}

function startRetrainPolling(validationData) {
  // Update steps as we get status
  updateProgressStep("stepValidate", "completed");
  updateProgressStep("stepClean", "active");
  
  if (retrainPollInterval) {
    clearInterval(retrainPollInterval);
  }

  retrainPollInterval = setInterval(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/retrain-status`);
      const status = await response.json();

      if (status.status === "running") {
        // Update progress based on status message
        if (status.progress >= 30) updateProgressStep("stepClean", "completed");
        if (status.progress >= 50) updateProgressStep("stepMerge", "active");
        if (status.progress >= 70) updateProgressStep("stepMerge", "completed");
        if (status.progress >= 80) updateProgressStep("stepTrain", "active");
        if (status.progress >= 90) updateProgressStep("stepTrain", "completed");
        
        document.getElementById("retrainSubmitBtn").textContent = `${status.message} (${status.progress}%)`;
      } 
      else if (status.status === "completed" || status.status === "completed_keep_old" || status.status === "failed") {
        clearInterval(retrainPollInterval);
        retrainPollInterval = null;
        
        // Update final step
        updateProgressStep("stepCompare", status.status === "failed" ? "" : "completed");
        
        // Show summary
        await showRetrainSummary(status);
        
        // Reset button
        const submitBtn = document.getElementById("retrainSubmitBtn");
        submitBtn.disabled = false;
        submitBtn.textContent = "🚀 Upload va Huan luyen lai Model";
      }
    } catch (error) {
      console.error("Polling error:", error);
    }
  }, 2000);
}

async function showRetrainSummary(status) {
  const summaryDiv = document.getElementById("retrainSummary");
  const statusCard = document.getElementById("retrainStatusCard");
  const statusTitle = document.getElementById("retrainStatusTitle");
  const statusMessage = document.getElementById("retrainStatusMessage");
  const detailsDiv = document.getElementById("retrainDetails");
  const comparisonTable = document.getElementById("comparisonTable");

  summaryDiv.hidden = false;

  if (status.status === "completed") {
    statusCard.className = "retrain-status-card success";
    statusTitle.textContent = "✅ Model moi da duoc trien khai!";
    statusMessage.textContent = "Model moi co the loai bo cac truong hop tieu duong tot hon model cu.";
    showSuccess("Huan luyen hoan tat! Model moi da duoc trien khai.");
  } else if (status.status === "completed_keep_old") {
    statusCard.className = "retrain-status-card keep-old";
    statusTitle.textContent = "⚠️ Model cu duoc giu lai";
    statusMessage.textContent = "Model moi khong tot hon model cu, nen khong cap nhat.";
  } else {
    statusCard.className = "retrain-status-card error";
    statusTitle.textContent = "❌ Huan luyen that bai";
    statusMessage.textContent = status.message || "Da xay ra loi trong qua trinh huan luyen.";
    showError("Huan luyen that bai: " + (status.message || "Loi khong xac dinh"));
    return;
  }

  // Get detailed comparison
  try {
    const response = await fetch(`${API_BASE_URL}/retrain-details`);
    const details = await response.json();

    if (details.comparison_summary) {
      detailsDiv.hidden = false;
      comparisonTable.innerHTML = `
        <table>
          <thead>
            <tr>
              <th>Chi so</th>
              <th>Model cu</th>
              <th>Model moi</th>
              <th>Thay doi</th>
            </tr>
          </thead>
          <tbody>
            ${Object.entries(details.comparison_summary).map(([metric, data]) => `
              <tr>
                <td>${metric.toUpperCase()}</td>
                <td>${(data.old * 100).toFixed(1)}%</td>
                <td>${(data.new * 100).toFixed(1)}%</td>
                <td class="${data.improved ? 'improved' : 'declined'}">
                  ${data.improved ? '↑' : '↓'} ${Math.abs(data.change_percent).toFixed(1)}%
                </td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      `;
    }
  } catch (e) {
    console.error("Could not load details:", e);
  }
}

// API Status check
async function checkApiStatus() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    const status = document.getElementById("apiStatus");
    if (response.ok && status) {
      status.textContent = "Backend: hoạt động";
      status.classList.add("ok");
      status.classList.remove("error");
    }
  } catch (e) {
    const status = document.getElementById("apiStatus");
    if (status) {
      status.textContent = "Backend: chưa kết nối";
      status.classList.remove("ok");
      status.classList.add("error");
    }
  }
}

// Initialize
checkApiStatus();
