const API_BASE_URL = "http://localhost:5000";

const MODEL_LABELS = {
  xgboost: "XGBoost",
};

const METRIC_LABELS = {
  roc_auc: "ROC-AUC",
  f1: "F1-score",
  recall: "Recall",
  precision: "Precision",
};

const MODEL_KEYS = ["xgboost"];
const METRIC_KEYS = ["roc_auc", "f1", "recall", "precision"];

let selectedFile = null;
let dailyChart = null;
let pollTimer = null;

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || data.details || data.hint || "Request failed");
  }
  return data;
}

function updateApiStatus(ok, text) {
  const status = document.querySelector("#apiStatus");
  if (!status) return;
  status.textContent = text;
  status.classList.toggle("ok", ok);
  status.classList.toggle("error", !ok);
}

function formatInt(value) {
  return Number(value || 0).toLocaleString("vi-VN");
}

function formatDate(value) {
  if (!value) return "-";
  return new Date(value).toLocaleString("vi-VN", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatMetric(value) {
  return typeof value === "number" ? value.toFixed(3) : "-";
}

function formatDiff(diff) {
  if (typeof diff !== "number") return "-";
  const sign = diff > 0 ? "+" : "";
  return `${sign}${diff.toFixed(3)}`;
}

function changeClass(diff) {
  if (diff < 0) return "declined";
  if (diff >= 0.01) return "improved";
  if (diff > 0) return "slightly-improved";
  return "";
}

function renderDailyChart(dailyCounts) {
  const canvas = document.querySelector("#dailyChart");
  if (!canvas || typeof Chart === "undefined") return;
  if (dailyChart) dailyChart.destroy();

  dailyChart = new Chart(canvas, {
    type: "line",
    data: {
      labels: dailyCounts.map((item) => item.date),
      datasets: [
        {
          label: "Dữ liệu mới",
          data: dailyCounts.map((item) => item.count),
          borderColor: "#0891B2",
          backgroundColor: "rgba(8, 145, 178, 0.12)",
          fill: true,
          tension: 0.3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
    },
  });
}

function renderConditions(stats) {
  const conditions = stats.conditions || {};
  const items = [
    {
      ok: conditions.enough_samples,
      label: "Đã có 500+ mẫu mới",
      value: `Hiện tại: ${formatInt(conditions.current_samples)} mẫu`,
    },
    {
      ok: conditions.has_two_classes,
      label: "Dữ liệu mới đa dạng cả 2 class",
      value: `Class 0: ${stats.class_dist?.["0"] || 0}, Class 1: ${stats.class_dist?.["1"] || 0}`,
    },
    {
      ok: conditions.days_since_last_retrain_ok,
      label: "Chưa retrain trong 30 ngày",
      value: conditions.days_since_last_retrain == null ? "Chưa có lịch sử retrain" : `${conditions.days_since_last_retrain} ngày`,
    },
  ];

  document.querySelector("#conditionList").innerHTML = items.map((item) => `
    <div class="condition-item ${item.ok ? "ok" : "warn"}">
      <span>${item.ok ? "✓" : "!"}</span>
      <div>
        <strong>${item.label}</strong>
        <small>${item.value}</small>
      </div>
    </div>
  `).join("");
}

function renderStats(stats) {
  document.querySelector("#totalOriginal").textContent = formatInt(stats.total_original);
  document.querySelector("#totalNew").textContent = formatInt(stats.total_new);
  document.querySelector("#totalCombined").textContent = formatInt(stats.total_combined);
  document.querySelector("#lastRetrain").textContent = formatDate(stats.last_retrain);

  const percent0 = stats.class_percent?.["0"] || 0;
  const percent1 = stats.class_percent?.["1"] || 0;
  document.querySelector("#classZero").textContent = `${percent0.toFixed(1)}%`;
  document.querySelector("#classOne").textContent = `${percent1.toFixed(1)}%`;
  document.querySelector("#classZeroBar").style.width = `${percent0}%`;
  document.querySelector("#classOneBar").style.width = `${percent1}%`;
  document.querySelector("#classHint").textContent = stats.total_labeled
    ? `Có ${formatInt(stats.total_labeled)} dòng có nhãn trong dữ liệu mới.`
    : "Chưa có dữ liệu nhãn để đánh giá cân bằng.";

  renderDailyChart(stats.daily_counts || []);
  renderConditions(stats);
}

function setProgress(progress, message) {
  const steps = [
    ["merge", 10],
    ["preprocess", 25],
    ["xgb", 60],
    ["eval", 85],
    ["done", 100],
  ];
  document.querySelector("#progressBlock").hidden = false;
  steps.forEach(([name, threshold]) => {
    const step = document.querySelector(`[data-step="${name}"]`);
    step.classList.toggle("completed", progress >= threshold);
    step.classList.toggle("active", progress < threshold && progress >= threshold - 15);
  });
  if (progress >= 100) {
    document.querySelector("#doneStepText").textContent = message || "Hoàn tất";
  }
}

function buildMetricCell(oldValue, newValue) {
  const diff = typeof oldValue === "number" && typeof newValue === "number" ? newValue - oldValue : null;
  return `
    <td>${formatMetric(oldValue)}</td>
    <td>${formatMetric(newValue)}</td>
    <td class="${changeClass(diff)}">${formatDiff(diff)}</td>
  `;
}

function renderComparison(details) {
  const oldMetrics = details.old_metrics || {};
  const newMetrics = details.new_metrics || {};

  document.querySelector("#comparisonTables").innerHTML = MODEL_KEYS.map((modelKey) => `
    <section class="model-comparison">
      <h3>${MODEL_LABELS[modelKey]}</h3>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Metrics</th>
              <th>Model cũ</th>
              <th>Model mới</th>
              <th>Thay đổi</th>
            </tr>
          </thead>
          <tbody>
            ${METRIC_KEYS.map((metric) => `
              <tr>
                <td>${METRIC_LABELS[metric]}</td>
                ${buildMetricCell(oldMetrics[modelKey]?.[metric], newMetrics[modelKey]?.[metric])}
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </section>
  `).join("");

  const blocked = details.full_comparison?.decision_summary?.blocked_models || [];
  const canDeploy = Boolean(details.can_deploy);
  const summary = canDeploy
    ? [
        "✓ XGBoost không giảm ở ROC-AUC, F1 và Recall",
        "✓ Có cải thiện đủ điều kiện ở chỉ số chính",
        "✓ Có thể cập nhật model mới",
      ]
    : [
        `⚠ XGBoost chưa đạt điều kiện`,
        "⚠ Không khuyến nghị cập nhật nếu Recall, F1 hoặc ROC-AUC giảm",
        "✗ Nên giữ model cũ cho đến khi dữ liệu mới tốt hơn",
      ];

  document.querySelector("#decisionSummary").className = `decision-summary ${canDeploy ? "success" : "warning"}`;
  document.querySelector("#decisionSummary").innerHTML = summary.map((line) => `<p>${line}</p>`).join("");
  document.querySelector("#applyBtn").hidden = !canDeploy;
  document.querySelector("#comparisonPanel").hidden = false;
}

async function loadStats() {
  const stats = await fetchJson(`${API_BASE_URL}/data-stats`);
  renderStats(stats);
  updateApiStatus(true, "Backend: hoạt động");
}

async function loadHistory() {
  const data = await fetchJson(`${API_BASE_URL}/retrain-history`);
  const history = data.history || [];
  document.querySelector("#historyBody").innerHTML = history.length ? history.map((item) => `
    <tr>
      <td>#${item.id}</td>
      <td>${formatDate(item.created_at)}</td>
      <td>${formatInt(item.new_rows)}</td>
      <td>${formatMetric(item.old_roc_auc)}</td>
      <td>${formatMetric(item.new_roc_auc)}</td>
      <td>${item.decision === "updated" ? "Đã cập nhật" : "Giữ cũ"}</td>
    </tr>
  `).join("") : `<tr><td colspan="6">Chưa có lịch sử retrain.</td></tr>`;
}

function setupUpload() {
  const input = document.querySelector("#csvInput");
  const dropZone = document.querySelector("#csvDropZone");
  const button = document.querySelector("#uploadBtn");

  function selectFile(file) {
    if (!file) return;
    if (!file.name.endsWith(".csv")) {
      document.querySelector("#uploadError").textContent = "Chỉ chấp nhận file CSV.";
      document.querySelector("#uploadError").hidden = false;
      return;
    }
    selectedFile = file;
    document.querySelector("#fileName").textContent = file.name;
    document.querySelector("#fileInfo").hidden = false;
    button.disabled = false;

    const reader = new FileReader();
    reader.onload = (event) => {
      const rows = Math.max(0, event.target.result.split("\n").filter(Boolean).length - 1);
      document.querySelector("#fileRows").textContent = `${formatInt(rows)} dòng`;
    };
    reader.readAsText(file);
  }

  input.addEventListener("change", (event) => selectFile(event.target.files[0]));
  dropZone.addEventListener("click", () => input.click());
  dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("drag-over");
    selectFile(event.dataTransfer.files[0]);
  });

  document.querySelector("#uploadForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!selectedFile) return;
    button.disabled = true;
    button.textContent = "Đang upload...";
    document.querySelector("#uploadError").hidden = true;
    document.querySelector("#uploadMessage").hidden = true;

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      const result = await fetchJson(`${API_BASE_URL}/upload-new-data`, { method: "POST", body: formData });
      document.querySelector("#uploadMessage").textContent = `Đã thêm ${formatInt(result.rows_added)} dòng vào dữ liệu mới.`;
      document.querySelector("#uploadMessage").hidden = false;
      selectedFile = null;
      input.value = "";
      document.querySelector("#fileInfo").hidden = true;
      await loadStats();
    } catch (error) {
      document.querySelector("#uploadError").textContent = error.message;
      document.querySelector("#uploadError").hidden = false;
    } finally {
      button.disabled = !selectedFile;
      button.textContent = "Thêm vào dữ liệu mới";
    }
  });
}

async function pollRetrainStatus() {
  const status = await fetchJson(`${API_BASE_URL}/retrain-status`);
  setProgress(status.progress || 0, status.message);

  if (status.status === "running") return;
  clearInterval(pollTimer);
  pollTimer = null;
  document.querySelector("#retrainBtn").disabled = false;

  if (status.status === "failed") {
    document.querySelector("#retrainHint").textContent = status.message || "Retrain thất bại.";
    return;
  }

  const details = await fetchJson(`${API_BASE_URL}/retrain-details`);
  renderComparison(details);
}

function setupRetrain() {
  document.querySelector("#retrainBtn").addEventListener("click", async () => {
    const button = document.querySelector("#retrainBtn");
    button.disabled = true;
    document.querySelector("#comparisonPanel").hidden = true;
    document.querySelector("#retrainHint").textContent = "Đang khởi động retrain...";
    setProgress(10, "Đang bắt đầu");

    try {
      await fetchJson(`${API_BASE_URL}/retrain`, { method: "POST" });
      pollTimer = setInterval(() => {
        pollRetrainStatus().catch((error) => {
          console.error(error);
          clearInterval(pollTimer);
          pollTimer = null;
          button.disabled = false;
        });
      }, 2000);
    } catch (error) {
      button.disabled = false;
      document.querySelector("#retrainHint").textContent = error.message;
    }
  });

  document.querySelector("#applyBtn").addEventListener("click", () => applyDecision(true));
  document.querySelector("#keepOldBtn").addEventListener("click", () => applyDecision(false));
}

async function applyDecision(confirm) {
  const applyBtn = document.querySelector("#applyBtn");
  const keepBtn = document.querySelector("#keepOldBtn");
  applyBtn.disabled = true;
  keepBtn.disabled = true;

  try {
    await fetchJson(`${API_BASE_URL}/apply-new-model`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm }),
    });
    document.querySelector("#retrainHint").textContent = confirm ? "Đã cập nhật model mới." : "Đã giữ model cũ.";
    document.querySelector("#comparisonPanel").hidden = true;
    await Promise.all([loadStats(), loadHistory()]);
  } catch (error) {
    document.querySelector("#retrainHint").textContent = error.message;
  } finally {
    applyBtn.disabled = false;
    keepBtn.disabled = false;
  }
}

async function init() {
  setupUpload();
  setupRetrain();
  try {
    await Promise.all([loadStats(), loadHistory()]);
  } catch (error) {
    updateApiStatus(false, "Backend: lỗi");
    console.error(error);
  }
}

init();
