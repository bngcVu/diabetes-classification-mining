const API_BASE_URL = "http://localhost:5000";

const MODEL_LABELS = {
  xgboost: "XGBoost",
};

const MODEL_COLORS = {
  xgboost: "#16a34a",
};

const METRIC_LABELS = {
  accuracy: "Accuracy",
  precision: "Precision",
  recall: "Recall",
  f1: "F1-Score",
  roc_auc: "ROC-AUC",
};

const charts = {};
let cachedStats = null;

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

async function loadStats() {
  return fetchJson(`${API_BASE_URL}/model-stats`);
}

function formatMetric(value) {
  return typeof value === "number" ? value.toFixed(3) : "-";
}

function getModelKeys(metrics) {
  return ["xgboost"].filter((key) => metrics[key]);
}

function renderMetricsTable(metrics) {
  const rows = getModelKeys(metrics).map((modelKey) => {
    const item = metrics[modelKey];
    return `
      <tr>
        <td>${MODEL_LABELS[modelKey]}</td>
        <td>${formatMetric(item.accuracy)}</td>
        <td>${formatMetric(item.precision)}</td>
        <td>${formatMetric(item.recall)}</td>
        <td>${formatMetric(item.f1)}</td>
        <td>${formatMetric(item.roc_auc)}</td>
      </tr>
    `;
  });
  document.querySelector("#metricsBody").innerHTML = rows.join("");
}

function destroyChart(id) {
  if (charts[id]) {
    charts[id].destroy();
    charts[id] = null;
  }
}

function renderMetricsChart(metrics) {
  const modelKeys = getModelKeys(metrics);
  const metricKeys = ["accuracy", "precision", "recall", "f1", "roc_auc"];

  destroyChart("metrics");
  charts.metrics = new Chart(document.querySelector("#metricsChart"), {
    type: "bar",
    data: {
      labels: metricKeys.map(k => METRIC_LABELS[k] || k),
      datasets: modelKeys.map((modelKey) => ({
        label: MODEL_LABELS[modelKey],
        data: metricKeys.map((metricKey) => metrics[modelKey][metricKey]),
        backgroundColor: MODEL_COLORS[modelKey],
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 0, max: 1 },
      },
      plugins: {
        title: {
          display: true,
          text: "Model Metrics Comparison"
        }
      }
    },
  });
}

function toRocPoints(roc) {
  return roc.fpr.map((fpr, index) => ({ x: fpr, y: roc.tpr[index] }));
}

function toPrPoints(pr) {
  return pr.recall.map((recall, index) => ({ x: recall, y: pr.precision[index] }));
}

function renderRocChart(curves) {
  const modelKeys = getModelKeys(curves);
  destroyChart("roc");
  charts.roc = new Chart(document.querySelector("#rocChart"), {
    type: "line",
    data: {
      datasets: [
        ...modelKeys.map((modelKey) => ({
          label: `${MODEL_LABELS[modelKey]} AUC = ${curves[modelKey].roc.auc.toFixed(3)}`,
          data: toRocPoints(curves[modelKey].roc),
          borderColor: MODEL_COLORS[modelKey],
          backgroundColor: "transparent",
          parsing: false,
          pointRadius: 0,
        })),
        {
          label: "Random",
          data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
          borderColor: "#9ca3af",
          backgroundColor: "transparent",
          borderDash: [6, 6],
          parsing: false,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { type: "linear", min: 0, max: 1, title: { display: true, text: "False Positive Rate (FPR)" } },
        y: { min: 0, max: 1, title: { display: true, text: "True Positive Rate (TPR)" } },
      },
    },
  });
}

function renderPrChart(curves) {
  const modelKeys = getModelKeys(curves);
  destroyChart("pr");
  charts.pr = new Chart(document.querySelector("#prChart"), {
    type: "line",
    data: {
      datasets: modelKeys.map((modelKey) => ({
        label: `${MODEL_LABELS[modelKey]} AP = ${curves[modelKey].precision_recall.average_precision.toFixed(3)}`,
        data: toPrPoints(curves[modelKey].precision_recall),
        borderColor: MODEL_COLORS[modelKey],
        backgroundColor: "transparent",
        parsing: false,
        pointRadius: 0,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { type: "linear", min: 0, max: 1, title: { display: true, text: "Recall" } },
        y: { min: 0, max: 1, title: { display: true, text: "Precision" } },
      },
    },
  });
}

function renderImportanceChart(modelKey) {
  if (!cachedStats || !cachedStats.feature_importance[modelKey]) {
    return;
  }

  const rows = cachedStats.feature_importance[modelKey].slice(0, 15).reverse();
  destroyChart("importance");
  charts.importance = new Chart(document.querySelector("#importanceChart"), {
    type: "bar",
    data: {
      labels: rows.map((item) => item.feature_vi || item.feature),
      datasets: [
        {
          label: MODEL_LABELS[modelKey],
          data: rows.map((item) => item.importance),
          backgroundColor: MODEL_COLORS[modelKey],
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y",
      plugins: {
        title: {
          display: true,
          text: "Feature Importance - " + MODEL_LABELS[modelKey]
        }
      }
    },
  });
}

function renderConfusionMatrices(metrics) {
  const html = getModelKeys(metrics).map((modelKey) => {
    const [[tn, fp], [fn, tp]] = metrics[modelKey].confusion_matrix;
    return `
      <article class="matrix-card">
        <h3>${MODEL_LABELS[modelKey]}</h3>
        <div class="confusion-grid">
          <div><span>TN (Đúng: Không TD)</span><strong>${tn.toLocaleString()}</strong></div>
          <div><span>FP (Sai: Báo nhầm)</span><strong>${fp.toLocaleString()}</strong></div>
          <div><span>FN (Sai: Bỏ sót)</span><strong>${fn.toLocaleString()}</strong></div>
          <div><span>TP (Đúng: Phát hiện)</span><strong>${tp.toLocaleString()}</strong></div>
        </div>
      </article>
    `;
  });
  document.querySelector("#confusionMatrices").innerHTML = html.join("");
}

function renderConfusionBarChart(metrics) {
  const modelKeys = getModelKeys(metrics);
  const confusionTypes = ["TP", "FN", "FP", "TN"];
  const getValue = (matrix, type) => {
    const [[tn, fp], [fn, tp]] = matrix;
    return { TN: tn, FP: fp, FN: fn, TP: tp }[type];
  };

  destroyChart("confusionBar");
  charts.confusionBar = new Chart(document.querySelector("#confusionBarChart"), {
    type: "bar",
    data: {
      labels: confusionTypes,
      datasets: modelKeys.map((modelKey) => ({
        label: MODEL_LABELS[modelKey],
        data: confusionTypes.map((type) => getValue(metrics[modelKey].confusion_matrix, type)),
        backgroundColor: MODEL_COLORS[modelKey],
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
    },
  });
}

function setupTabs() {
  const buttons = document.querySelectorAll(".tab-button");
  const contents = {
    metrics: document.querySelector("#metricsTab"),
    roc: document.querySelector("#rocTab"),
    pr: document.querySelector("#prTab"),
    importance: document.querySelector("#importanceTab"),
    confusion: document.querySelector("#confusionTab"),
  };

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const tab = button.dataset.tab;
      buttons.forEach((item) => item.classList.remove("active"));
      Object.values(contents).forEach((content) => content.classList.remove("active"));
      button.classList.add("active");
      contents[tab].classList.add("active");
    });
  });
}

function setupImportanceToggle() {
  document.querySelectorAll(".segment-button").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".segment-button").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      renderImportanceChart(button.dataset.importanceModel);
    });
  });
}

async function init() {
  setupTabs();

  try {
    cachedStats = await loadStats();
    renderMetricsTable(cachedStats.metrics);
    renderMetricsChart(cachedStats.metrics);
    renderRocChart(cachedStats.curves);
    renderPrChart(cachedStats.curves);
    renderImportanceChart("xgboost");
    renderConfusionMatrices(cachedStats.metrics);
    renderConfusionBarChart(cachedStats.metrics);
  } catch (error) {
    console.error(error);
    alert("Khong the tai du lieu tu backend. Dam bao backend dang chay tai localhost:5000");
  }
}

init();
