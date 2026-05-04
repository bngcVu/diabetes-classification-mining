const API_BASE_URL = "http://localhost:5000";

const MODEL_LABELS = {
  xgboost: "XGBoost",
};

const MODEL_COLORS = {
  xgboost: "#16a34a",
};

const METRIC_LABELS = {
  accuracy: "Độ chính xác",
  precision: "Precision",
  recall: "Recall",
  f1: "F1-Score",
  roc_auc: "ROC-AUC",
};

const METRIC_EXPLANATIONS = {
  accuracy: {
    short: "Tỷ lệ dự đoán đúng trên tổng số",
    explanation: "Cho biết bao nhiêu phần trăm dự đoán là chính xác (đúng/tổng). Ví dụ: 75% accuracy nghĩa là 75/100 dự đoán là đúng."
  },
  precision: {
    short: "Độ chính xác khi dự đoán dương tính",
    explanation: "Trong số những người dự đoán bị tiểu đường, bao nhiêu thật sự bị. Ví dụ: 70% precision = 70% trong số ai được dự đoán bị tiểu đường là đúng."
  },
  recall: {
    short: "Tỷ lệ phát hiện người bị tiểu đường",
    explanation: "Trong số những người thật sự bị tiểu đường, bao nhiêu được phát hiện. Ví dụ: 80% recall = phát hiện được 80% những người bị tiểu đường."
  },
  f1: {
    short: "Cân bằng giữa precision và recall",
    explanation: "Trung bình diện tích cân bằng giữa precision và recall. Chỉ số F1 cao nghĩa là model tốt cả hai bên."
  },
  roc_auc: {
    short: "Khả năng phân biệt giữa các lớp",
    explanation: "Điểm 1.0 là phân biệt hoàn hảo, 0.5 là ngẫu nhiên. Ví dụ: 0.85 AUC nghĩa là 85% khả năng phân biệt đúng giữa người bị và không bị tiểu đường."
  }
};

const FEATURE_EXPLANATIONS = {
  "HighBP": {
    name: "Huyết áp cao",
    what: "0 = Không, 1 = Có (huyết áp tâm thu >= 140 mmHg)",
    description: "Huyết áp cao là yếu tố nguy cơ hàng đầu của bệnh tiểu đường type 2. Huyết áp tâm thu từ 140 mmHg trở lên được coi là cao.",
    tip: "Giữ huyết áp dưới 130/80 mmHg bằng cách giảm muối, tập thể dục đều đặn."
  },
  "HighChol": {
    name: "Cholesterol cao",
    what: "0 = Không, 1 = Có (cholesterol xấu LDL >= 3.4 mmol/L)",
    description: "Cholesterol LDL (xấu) cao gây tích tụ mỡ trong mạch máu, tăng nguy cơ tiểu đường và bệnh tim mạch.",
    tip: "Hạn chế thức ăn nhiều cholesterol, tăng cường rau xanh và ngũ cốc nguyên hạt."
  },
  "CholCheck": {
    name: "Kiểm tra Cholesterol",
    what: "0 = Không, 1 = Có (đã kiểm tra trong 5 năm qua)",
    description: "Việc kiểm tra cholesterol định kỳ giúp phát hiện sớm các vấn đề về mỡ máu trước khi gây biến chứng.",
    tip: "Nên kiểm tra cholesterol ít nhất 1 lần/năm nếu trên 20 tuổi."
  },
  "BMI": {
    name: "Chỉ số BMI",
    what: "Từ 10 - 80 (BMI = Cân nặng / Chiều cao²)",
    description: "BMI = Cân nặng (kg) / Chiều cao² (m²). BMI từ 25-29.9 là thừa cân, từ 30 trở lên là béo phì - đây là yếu tố nguy cơ lớn nhất của tiểu đường type 2.",
    tip: "BMI lý tưởng cho người châu Á: 18.5-22.9. Giảm 5-10% cân nặng có thể giảm 50% nguy cơ tiểu đường."
  },
  "Smoker": {
    name: "Hút thuốc",
    what: "0 = Không hút/Đã bỏ, 1 = Đã hút ít nhất 100 điếu thuốc",
    description: "Hút thuốc lá làm tăng đề kháng insulin và viêm mạch máu, làm tăng nguy cơ tiểu đường type 2 lên 30-40%.",
    tip: "Bỏ thuốc lá hoàn toàn có thể giảm nguy cơ tiểu đường về mức như người chưa từng hút."
  },
  "Stroke": {
    name: "Đột quỵ",
    what: "0 = Chưa từng, 1 = Đã từng bị đột quỵ",
    description: "Đột quỵ xảy ra khi máu không đến não được. Tiểu đường làm tăng nguy cơ đột quỵ gấp 2-4 lần.",
    tip: "Kiểm soát đường huyết, huyết áp và cholesterol là cách tốt nhất để phòng ngừa đột quỵ."
  },
  "HeartDiseaseorAttack": {
    name: "Bệnh tim",
    what: "0 = Không, 1 = Đã từng bị bệnh tim hoặc nhồi máu cơ tim",
    description: "Bệnh tim mạch vành và nhồi máu cơ tim có liên quan chặt chẽ với tiểu đường do tổn thương mạch máu.",
    tip: "Tiểu đường và bệnh tim thường đi cùng nhau. Cần kiểm soát cả đường huyết lẫn sức khỏe tim mạch."
  },
  "PhysActivity": {
    name: "Hoạt động thể chất",
    what: "0 = Không, 1 = Có tập thể dục trong 30 ngày qua",
    description: "Tập thể dục đều đặn giúp cơ thể sử dụng insulin hiệu quả hơn, giảm đề kháng insulin - nguyên nhân chính của tiểu đường type 2.",
    tip: "Đi bộ nhanh 30 phút/ngày hoặc 150 phút/tuần có thể giảm nguy cơ tiểu đường tới 30%."
  },
  "Fruits": {
    name: "Ăn trái cây",
    what: "0 = Không ăn đều, 1 = Ăn trái cây mỗi ngày trong 30 ngày qua",
    description: "Trái cây chứa nhiều vitamin, khoáng chất và chất xơ tốt cho sức khỏe. Tuy nhiên cần chọn loại ít đường và ăn vừa phải.",
    tip: "Nên ăn 2-3 phần trái cây/ngày. Ưu tiên táo, cam, berries thay vì xoài, nho nhiều đường."
  },
  "Veggies": {
    name: "Ăn rau",
    what: "0 = Không ăn đều, 1 = Ăn rau xanh mỗi ngày trong 30 ngày qua",
    description: "Rau xanh chứa ít đường, nhiều chất xơ và vi chất dinh dưỡng giúp kiểm soát đường huyết hiệu quả.",
    tip: "Nên ăn ít nhất 2-3 chén rau xanh/ngày. Đặc biệt tốt: bông cải xanh, rau bina, cải bó xôi."
  },
  "HvyAlcoholConsump": {
    name: "Uống rượu nhiều",
    what: "0 = Không/uống ít, 1 = Uống nhiều (nam >14 ly/tuần, nữ >7 ly/tuần)",
    description: "Uống nhiều rượu làm tăng nguy cơ viêm tụy và rối loạn chuyển hóa đường.",
    tip: "Nếu uống rượu, hãy uống có chừng mực và luôn ăn kèm thức ăn để tránh hạ đường huyết đột ngột."
  },
  "AnyHealthcare": {
    name: "Có bảo hiểm y tế",
    what: "0 = Không có, 1 = Có bảo hiểm y tế hoặc bảo hiểm sức khỏe",
    description: "Người có bảo hiểm y tế thường được khám sức khỏe định kỳ và phát hiện tiểu đường sớm hơn.",
    tip: "Khám sức khỏe định kỳ 1 lần/năm giúp phát hiện tiểu đường sớm, dễ kiểm soát hơn."
  },
  "NoDocbcCost": {
    name: "Không khám vì chi phí",
    what: "0 = Không, 1 = Từng không đi khám bệnh vì chi phí trong 12 tháng",
    description: "Việc trì hoãn khám chữa bệnh vì lo ngại chi phí có thể dẫn đến phát hiện muộn và biến chứng nặng hơn.",
    tip: "Tiểu đường nếu phát hiện sớm và kiểm soát tốt sẽ tiết kiệm chi phí điều trị lâu dài."
  },
  "GenHlth": {
    name: "Sức khỏe tổng quát",
    what: "1 = Tuyệt vời, 2 = Rất tốt, 3 = Tốt, 4 = Kém, 5 = Rất kém",
    description: "Tự đánh giá sức khỏe tổng thể. Mức 1-2: Sức khỏe tốt, 3: Trung bình, 4-5: Sức khỏe kém. Người tự đánh giá sức khỏe kém có nguy cơ tiểu đường cao hơn.",
    tip: "Cải thiện lối sống: ăn uống lành mạnh, tập thể dục, ngủ đủ giấc để nâng cao sức khỏe tổng thể."
  },
  "MentHlth": {
    name: "Sức khỏe tinh thần",
    what: "0 - 30 ngày: Số ngày sức khỏe tinh thần không tốt (stress, lo âu, trầm buồn) trong 30 ngày qua",
    description: "Căng thẳng kéo dài và các vấn đề tâm lý làm tăng hormone cortisol, gây tăng đường huyết và đề kháng insulin. 0 = 30 ngày đều khỏe, 30 = 30 ngày đều không khỏe.",
    tip: "Thực hành thiền, yoga, hoặc các bài tập thở để giảm stress. Ngủ đủ 7-8 tiếng/đêm."
  },
  "PhysHlth": {
    name: "Sức khỏe thể chất",
    what: "0 - 30 ngày: Số ngày sức khỏe thể chất không tốt (đau ốm, chấn thương) trong 30 ngày qua",
    description: "Sức khỏe thể chất kém (đau ốm, chấn thương) ảnh hưởng đến khả năng vận động và kiểm soát đường huyết. 0 = 30 ngày đều khỏe, 30 = 30 ngày đều không khỏe.",
    tip: "Duy trì vận động nhẹ nhàng ngay cả khi không khỏe. Hỏi bác sĩ về bài tập phù hợp với tình trạng."
  },
  "DiffWalk": {
    name: "Khó đi lại",
    what: "0 = Không gặp khó khăn, 1 = Gặp khó khăn khi đi bộ hoặc leo cầu thang",
    description: "Khó khăn khi đi lại có thể do béo phì, đau khớp, hoặc các vấn đề tuần hoàn - tất cả đều liên quan đến nguy cơ tiểu đường.",
    tip: "Bắt đầu với các bài tập nhẹ như bơi lội hoặc đạp xe để giảm áp lực lên khớp."
  },
  "Sex": {
    name: "Giới tính",
    what: "0 = Nữ, 1 = Nam",
    description: "Nam giới có nguy cơ tiểu đường type 2 cao hơn phụ nữ (do tỷ lệ mỡ nội tạng cao hơn), nhưng phụ nữ có nguy cơ biến chứng tim mạch cao hơn khi mắc tiểu đường.",
    tip: "Cả nam và nữ đều cần chú ý phòng ngừa tiểu đường bằng lối sống lành mạnh."
  },
  "Age": {
    name: "Nhóm tuổi",
    what: "1 = 18-24 tuổi → 13 = 80 tuổi trở lên (theo thang 13 nhóm tuổi)",
    description: "Nguy cơ tiểu đường tăng theo tuổi, đặc biệt sau 45 tuổi. Sau 65 tuổi, khoảng 25% người bị tiểu đường.",
    tip: "Sau 45 tuổi nên xét nghiệm đường huyết định kỳ mỗi năm, dù không có triệu chứng."
  },
  "Education": {
    name: "Học vấn",
    what: "1 = Không đi học → 6 = Sau đại học (Thạc sĩ, Tiến sĩ)",
    description: "Học vấn liên quan đến nhận thức sức khỏe: người học cao thường hiểu rõ hơn về dinh dưỡng, tầm quan trọng của tập thể dục, và đi khám sức khỏe định kỳ.",
    tip: "Dù ở mức học vấn nào, việc tìm hiểu về tiểu đường và phòng ngừa đều rất quan trọng."
  },
  "Income": {
    name: "Thu nhập",
    what: "1 = < $10,000/năm → 8 = > $75,000/năm (8 mức thu nhập)",
    description: "Thu nhập ảnh hưởng trực tiếp đến lối sống và sức khỏe: người thu nhập thấp thường khó tiếp cận thực phẩm lành mạnh, ít thời gian tập thể dục, và hạn chế khám bệnh.",
    tip: "Ăn uống lành mạnh không nhất thiết phải tốn kém: rau xanh, đậu, ngũ cốc nguyên hạt đều rẻ và tốt cho sức khỏe."
  }
};

const FEATURE_LABELS_VI = {
  "HighBP": "Huyết áp cao (0/1)",
  "HighChol": "Cholesterol cao (0/1)",
  "CholCheck": "Kiểm tra Cholesterol (0/1)",
  "BMI": "Chỉ số BMI",
  "Smoker": "Hút thuốc (0/1)",
  "Stroke": "Đột quỵ (0/1)",
  "HeartDiseaseorAttack": "Bệnh tim (0/1)",
  "PhysActivity": "Hoạt động thể chất (0/1)",
  "Fruits": "Ăn trái cây (0/1)",
  "Veggies": "Ăn rau (0/1)",
  "HvyAlcoholConsump": "Uống rượu nhiều (0/1)",
  "AnyHealthcare": "Có bảo hiểm y tế (0/1)",
  "NoDocbcCost": "Không khám vì chi phí (0/1)",
  "GenHlth": "Sức khỏe tổng quát (1-5)",
  "MentHlth": "Ngày sức khỏe tâm thần kém (0-30 ngày)",
  "PhysHlth": "Ngày sức khỏe thể chất kém (0-30 ngày)",
  "DiffWalk": "Khó đi lại (0/1)",
  "Sex": "Giới tính (0: Nữ, 1: Nam)",
  "Age": "Nhóm tuổi (1-13)",
  "Education": "Trình độ học vấn (1-6)",
  "Income": "Thu nhập hàng năm (1-8)",
  "comorbidity_score": "Điểm bệnh đi kèm",
  "healthy_lifestyle": "Lối sống lành mạnh",
};

const BINARY_FIELDS = [
  "HighBP",
  "HighChol",
  "CholCheck",
  "Smoker",
  "Stroke",
  "HeartDiseaseorAttack",
  "PhysActivity",
  "Fruits",
  "Veggies",
  "HvyAlcoholConsump",
  "AnyHealthcare",
  "NoDocbcCost",
  "DiffWalk",
  "Sex",
];

const NUMBER_FIELDS = {
  BMI: { min: 10, max: 80, value: 28, placeholder: "10 - 80" },
  MentHlth: { min: 0, max: 30, value: 0, placeholder: "0 - 30 ngày" },
  PhysHlth: { min: 0, max: 30, value: 0, placeholder: "0 - 30 ngày" },
};

const ORDINAL_FIELDS = {
  GenHlth: { min: 1, max: 5, value: 3, labels: {1: "Tuyệt vời", 2: "Rất tốt", 3: "Tốt", 4: "Kém", 5: "Rất kém"} },
  Age: { min: 1, max: 13, value: 8, labels: {1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80+"} },
  Education: { min: 1, max: 6, value: 5, labels: {1: "Không đi học", 2: "Tiểu học", 3: "THCS", 4: "THPT", 5: "Cao đẳng/ĐH", 6: "Sau ĐH"} },
  Income: { min: 1, max: 8, value: 6, labels: {1: "< $10,000", 2: "$10-15K", 3: "$15-20K", 4: "$20-25K", 5: "$25-35K", 6: "$35-50K", 7: "$50-75K", 8: "> $75,000"} },
};

const FEATURE_ORDER = [
  "HighBP",
  "HighChol",
  "CholCheck",
  "BMI",
  "Smoker",
  "Stroke",
  "HeartDiseaseorAttack",
  "PhysActivity",
  "Fruits",
  "Veggies",
  "HvyAlcoholConsump",
  "AnyHealthcare",
  "NoDocbcCost",
  "GenHlth",
  "MentHlth",
  "PhysHlth",
  "DiffWalk",
  "Sex",
  "Age",
  "Education",
  "Income",
];

const charts = {};
let cachedStats = null;

function renderFeatureFields() {
  const container = document.querySelector("#featureFields");
  if (!container) return;

  container.innerHTML = FEATURE_ORDER.map((field) => {
    if (BINARY_FIELDS.includes(field)) {
      const zeroLabel = field === "Sex" ? "Nữ" : "Không";
      const oneLabel = field === "Sex" ? "Nam" : "Có";
      const defaultValue = ["CholCheck", "PhysActivity", "Fruits", "Veggies", "AnyHealthcare"].includes(field)
        ? "1"
        : "0";

      return `
        <label>
          ${FEATURE_LABELS_VI[field] || field}
          <select name="${field}">
            <option value="0" ${defaultValue === "0" ? "selected" : ""}>${zeroLabel}</option>
            <option value="1" ${defaultValue === "1" ? "selected" : ""}>${oneLabel}</option>
          </select>
        </label>
      `;
    }

    if (NUMBER_FIELDS[field]) {
      const config = NUMBER_FIELDS[field];
      return `
        <label>
          ${FEATURE_LABELS_VI[field] || field}
          <input type="number" name="${field}" min="${config.min}" max="${config.max}" value="${config.value}" placeholder="${config.placeholder || ''}" required />
        </label>
      `;
    }

    const config = ORDINAL_FIELDS[field];
    const options = Array.from({ length: config.max - config.min + 1 }, (_, index) => config.min + index)
      .map((value) => {
        const label = config.labels ? config.labels[value] || value : value;
        const selected = value === config.value ? "selected" : "";
        return `<option value="${value}" ${selected}>${value} - ${label}</option>`;
      })
      .join("");

    return `
      <label>
        ${FEATURE_LABELS_VI[field] || field}
        <select name="${field}">
          ${options}
        </select>
      </label>
    `;
  }).join("");
}

function buildPayload(form) {
  const formData = new FormData(form);
  const features = {};

  FEATURE_ORDER.forEach((field) => {
    features[field] = Number(formData.get(field));
  });

  return {
    model: formData.get("model"),
    allow_save: formData.get("allow_save") === "on",
    features,
  };
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed");
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

async function predict(payload) {
  return fetchJson(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function renderPredictionResult(result) {
  const percent = Math.round(result.probability * 100);
  const probabilityText = document.querySelector("#probabilityText");
  const predictionLabel = document.querySelector("#predictionLabel");
  const probabilityBar = document.querySelector("#probabilityBar");
  const resultEmpty = document.querySelector("#resultEmpty");
  const resultContent = document.querySelector("#resultContent");
  const selectedModelText = document.querySelector("#selectedModelText");
  const probabilityValue = document.querySelector("#probabilityValue");
  const labelValue = document.querySelector("#labelValue");

  if (!resultContent) return;

  // Hide empty state, show content
  if (resultEmpty) resultEmpty.hidden = true;
  resultContent.hidden = false;

  // Update values
  probabilityText.textContent = `${percent}%`;
  predictionLabel.textContent = result.label_text;
  selectedModelText.textContent = result.model_display_name || MODEL_LABELS[result.model] || result.model;
  probabilityValue.textContent = (result.probability * 100).toFixed(2) + "%";
  labelValue.textContent = result.label === 1 ? "Nguy cơ cao" : "Nguy cơ thấp";

  // Update bar with color
  probabilityBar.style.width = `${percent}%`;
  probabilityBar.classList.remove("high", "very-high");
  if (percent >= 70) {
    probabilityBar.classList.add("very-high");
  } else if (percent >= 50) {
    probabilityBar.classList.add("high");
  }

  // Color the probability text
  probabilityText.style.color = percent >= 70 ? "var(--danger)" : percent >= 50 ? "var(--warning)" : "var(--success)";
}

async function handleSubmit(event) {
  event.preventDefault();

  const form = event.currentTarget;
  const button = form.querySelector("button[type='submit']");
  const errorElement = document.querySelector("#formError");

  try {
    errorElement.hidden = true;
    button.disabled = true;
    button.textContent = "Đang xử lý...";
    const result = await predict(buildPayload(form));
    renderPredictionResult(result);
    updateApiStatus(true, "Backend: hoạt động");
  } catch (error) {
    errorElement.textContent = error.message;
    errorElement.hidden = false;
    updateApiStatus(false, "Backend: lỗi request");
  } finally {
    button.disabled = false;
    button.textContent = "Dự đoán ngay";
  }
}

async function loadStats() {
  return fetchJson(`${API_BASE_URL}/model-stats`);
}

function formatMetric(value) {
  return typeof value === "number" ? (value * 100).toFixed(1) + "%" : "-";
}

function getModelKeys(metrics) {
  return ["logistic_regression", "random_forest", "xgboost"].filter((key) => metrics[key]);
}

function renderModelMetrics(modelKey) {
  const metrics = cachedStats?.metrics?.[modelKey];
  if (!metrics) return "";

  const metricKeys = ["accuracy", "precision", "recall", "f1", "roc_auc"];
  const highlightKey = "f1";

  return `
    <div class="metrics-grid">
      ${metricKeys.map(key => `
        <div class="metric-card ${key === highlightKey ? 'highlight' : ''}">
          <div class="metric-value">${formatMetric(metrics[key])}</div>
          <div class="metric-name">${METRIC_LABELS[key]}</div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderConfusionMatrixForModel(modelKey) {
  const metrics = cachedStats?.metrics?.[modelKey];
  if (!metrics || !metrics.confusion_matrix) return "";

  const [[tn, fp], [fn, tp]] = metrics.confusion_matrix;
  const total = tn + fp + fn + tp;

  return `
    <div class="confusion-display">
      <div class="confusion-item positive">
        <div class="confusion-label">True Positive (TP)</div>
        <div class="confusion-value">${tp.toLocaleString()}</div>
        <div class="confusion-desc">Đúng: phát hiện bệnh</div>
      </div>
      <div class="confusion-item negative">
        <div class="confusion-label">True Negative (TN)</div>
        <div class="confusion-value">${tn.toLocaleString()}</div>
        <div class="confusion-desc">Đúng: không bệnh</div>
      </div>
      <div class="confusion-item negative">
        <div class="confusion-label">False Negative (FN)</div>
        <div class="confusion-value">${fn.toLocaleString()}</div>
        <div class="confusion-desc">Bỏ sót: có bệnh nhưng không phát hiện</div>
      </div>
      <div class="confusion-item positive">
        <div class="confusion-label">False Positive (FP)</div>
        <div class="confusion-value">${fp.toLocaleString()}</div>
        <div class="confusion-desc">Nhầm: không bệnh nhưng báo có</div>
      </div>
    </div>
  `;
}

function renderMetricExplanations() {
  const container = document.querySelector("#metricExplanations");
  if (!container) return;

  const modelKey = document.querySelector("#metricsModelSelect")?.value || "xgboost";
  const metrics = cachedStats?.metrics?.[modelKey];
  if (!metrics) return;

  container.innerHTML = Object.entries(METRIC_EXPLANATIONS).map(([key, exp]) => {
    const value = metrics[key];
    const valueFormatted = (value * 100).toFixed(1) + "%";
    const icon = key === "recall" ? "!" : key === "precision" ? "?" : "i";

    return `
      <div class="explanation-item">
        <div class="explanation-icon">${icon}</div>
        <div class="explanation-text">
          <strong>${exp.short}:</strong> ${exp.explanation}
          <br><em>Model hien tai: ${valueFormatted}</em>
        </div>
      </div>
    `;
  }).join('');
}

function renderFeatureExplanationCards() {
  const container = document.querySelector("#featureCards");
  if (!container) return;

  container.innerHTML = FEATURE_ORDER.map(field => {
    const exp = FEATURE_EXPLANATIONS[field];
    if (!exp) return "";

    return `
      <div class="feature-card" onclick="this.classList.toggle('expanded')">
        <div class="feature-card-title">${exp.name}</div>
        <div class="feature-card-brief">${exp.what}</div>
        <div class="feature-card-detail">
          <p>${exp.description}</p>
          <div class="feature-card-tip">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
            <span>Loi khuyen: ${exp.tip}</span>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

async function loadAndRenderModelStats() {
  const metricsContent = document.querySelector("#metricsContent");
  const metricsGrid = document.querySelector("#metricsGrid");
  const confusionDisplay = document.querySelector("#confusionDisplay");
  const metricsSelect = document.querySelector("#metricsModelSelect");

  if (!cachedStats || !cachedStats.metrics) {
    try {
      cachedStats = await loadStats();
    } catch (e) {
      console.log("Chua co du lieu model stats");
      return;
    }
  }

  if (!cachedStats.metrics) return;

  // Show metrics content (always visible now)
  if (metricsContent) metricsContent.hidden = false;

  // Get selected model
  const selectedModel = metricsSelect?.value || "xgboost";

  // Render metrics
  if (metricsGrid) {
    metricsGrid.innerHTML = renderModelMetrics(selectedModel);
  }

  // Render confusion matrix
  if (confusionDisplay) {
    confusionDisplay.innerHTML = renderConfusionMatrixForModel(selectedModel);
  }

  // Render explanations
  renderMetricExplanations();
}

function setupModelSelectListener() {
  const metricsSelect = document.querySelector("#metricsModelSelect");
  if (!metricsSelect) return;

  // Load stats on page load and when model changes
  loadAndRenderModelStats();

  metricsSelect.addEventListener("change", () => {
    loadAndRenderModelStats();
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
      if (contents[tab]) contents[tab].classList.add("active");
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

function renderMetricsTable(metrics) {
  const body = document.querySelector("#metricsBody");
  if (!body) return;

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
  body.innerHTML = rows.join("");
}

function destroyChart(id) {
  if (charts[id]) {
    charts[id].destroy();
    charts[id] = null;
  }
}

function renderMetricsChart(metrics) {
  const canvas = document.querySelector("#metricsChart");
  if (!canvas || !cachedStats) return;

  const modelKeys = getModelKeys(metrics);
  const metricKeys = ["accuracy", "precision", "recall", "f1", "roc_auc"];

  destroyChart("metrics");
  charts.metrics = new Chart(canvas, {
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
          text: "So sanh chi so giua cac mo hinh"
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
  const canvas = document.querySelector("#rocChart");
  if (!canvas || !cachedStats) return;

  const modelKeys = getModelKeys(curves);
  destroyChart("roc");
  charts.roc = new Chart(canvas, {
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
          label: "Ngau nhien",
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
        x: { type: "linear", min: 0, max: 1, title: { display: true, text: "Ty le am duong (FPR)" } },
        y: { min: 0, max: 1, title: { display: true, text: "Ty le duong tinh (TPR)" } },
      },
    },
  });
}

function renderPrChart(curves) {
  const canvas = document.querySelector("#prChart");
  if (!canvas || !cachedStats) return;

  const modelKeys = getModelKeys(curves);
  destroyChart("pr");
  charts.pr = new Chart(canvas, {
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
        x: { type: "linear", min: 0, max: 1, title: { display: true, text: "Recall (Phat hien)" } },
        y: { min: 0, max: 1, title: { display: true, text: "Precision (Chinh xac)" } },
      },
    },
  });
}

function renderImportanceChart(modelKey) {
  const canvas = document.querySelector("#importanceChart");
  if (!canvas || !cachedStats || !cachedStats.feature_importance?.[modelKey]) {
    return;
  }

  const rows = cachedStats.feature_importance[modelKey].slice(0, 15).reverse();
  destroyChart("importance");
  charts.importance = new Chart(canvas, {
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
          text: "Do quan trong dac trung - " + MODEL_LABELS[modelKey]
        }
      }
    },
  });
}

function renderConfusionMatrices(metrics) {
  const container = document.querySelector("#confusionMatrices");
  if (!container) return;

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
  container.innerHTML = html.join("");
}

function renderConfusionBarChart(metrics) {
  const canvas = document.querySelector("#confusionBarChart");
  if (!canvas || !cachedStats) return;

  const modelKeys = getModelKeys(metrics);
  const confusionTypes = ["TP", "FN", "FP", "TN"];
  const getValue = (matrix, type) => {
    const [[tn, fp], [fn, tp]] = matrix;
    return { TN: tn, FP: fp, FN: fn, TP: tp }[type];
  };

  destroyChart("confusionBar");
  charts.confusionBar = new Chart(canvas, {
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

async function init() {
  renderFeatureFields();
  setupTabs();
  setupImportanceToggle();
  setupModelSelectListener();
  document.querySelector("#predictionForm")?.addEventListener("submit", handleSubmit);

  try {
    cachedStats = await loadStats();
    updateApiStatus(true, "Backend: hoạt động");
    renderMetricsTable(cachedStats.metrics);
    renderMetricsChart(cachedStats.metrics);
    renderRocChart(cachedStats.curves);
    renderPrChart(cachedStats.curves);
    renderImportanceChart("random_forest");
    renderConfusionMatrices(cachedStats.metrics);
    renderConfusionBarChart(cachedStats.metrics);
    loadAndRenderModelStats();
    renderFeatureExplanationCards();
  } catch (error) {
    updateApiStatus(false, "Backend: chưa có dữ liệu");
    console.error(error);
  }
}

init();
