function prepareFeatures(raw, configuration) {
  const missing = value => value === null || value === undefined || value === "";
  const values = { ...raw };
  const age = missing(raw.age) ? null : Number(raw.age);
  const glucose = missing(raw.avg_glucose_level) ? null : Number(raw.avg_glucose_level);
  const bmi = missing(raw.bmi) ? null : Number(raw.bmi);
  values.age_squared = age === null ? null : age ** 2;
  values.avg_glucose_log1p = glucose === null ? null : Math.log1p(Math.max(0, glucose));
  values.bmi_squared = bmi === null ? null : bmi ** 2;
  values.age_glucose_interaction = age === null || glucose === null ? null : age * glucose;
  values.bmi_glucose_interaction = bmi === null || glucose === null ? null : bmi * glucose;
  values.comorbidity_count = missing(raw.hypertension) || missing(raw.heart_disease)
    ? null : Number(raw.hypertension) + Number(raw.heart_disease);
  const numeric = configuration.numeric;
  const missingMask = numeric.columns.map(name => missing(values[name]));
  const encoded = numeric.columns.map((name, index) =>
    missingMask[index] ? numeric.medians[index] : Number(values[name]));
  encoded.push(...numeric.missing_indicators.map(index => Number(missingMask[index])));
  const scaled = encoded.map((value, index) => (value - numeric.mean[index]) / numeric.scale[index]);
  configuration.categorical.columns.forEach((name, index) => {
    const value = missing(values[name]) ? configuration.categorical.fill_values[index] : values[name];
    scaled.push(...configuration.categorical.categories[index].map(category => Number(value === category)));
  });
  return scaled;
}

function predictStroke(raw, configuration) {
  const features = prepareFeatures(raw, configuration);
  const model = configuration.model;
  let score;
  if (model.type === "random_forest") {
    const vector = features.map(Math.fround);
    score = model.trees.reduce((total, tree) => {
      let node = 0;
      while (tree.left[node] !== -1) {
        node = vector[tree.feature[node]] <= tree.threshold[node] ? tree.left[node] : tree.right[node];
      }
      return total + tree.score[node];
    }, 0) / model.trees.length;
  } else if (model.type === "logistic_regression") {
    const logit = model.coefficients.reduce((total, coefficient, index) => total + coefficient * features[index], model.intercept);
    score = logit >= 0 ? 1 / (1 + Math.exp(-logit)) : Math.exp(logit) / (1 + Math.exp(logit));
  } else {
    throw new Error("Unsupported model");
  }
  return { score, prediction: Number(score >= configuration.threshold) };
}

if (typeof module !== "undefined") {
  module.exports = { prepareFeatures, predictStroke };
}

if (typeof document !== "undefined") {
  const form = document.querySelector("form");
  const status = document.getElementById("status");
  const submit = document.getElementById("predict");
  const result = document.getElementById("result");
  const examples = {
    a: { age: 25, hypertension: 0, heart_disease: 0, ever_married: "No", avg_glucose_level: 85, bmi: 22, gender: "Female", work_type: "Private", residence_type: "Urban", smoking_status: "never smoked" },
    b: { age: 75, hypertension: 1, heart_disease: 1, ever_married: "Yes", avg_glucose_level: 220, bmi: 32, gender: "Male", work_type: "Self-employed", residence_type: "Rural", smoking_status: "formerly smoked" },
    missing: { age: 55, hypertension: 1, heart_disease: 0, ever_married: "Yes", avg_glucose_level: 145, bmi: "", gender: "Female", work_type: "Private", residence_type: "Urban", smoking_status: "Unknown" },
  };
  let configuration;
  form.addEventListener("input", () => {
    result.hidden = true;
    status.textContent = "กรอกข้อมูลแล้วกดประเมินได้เลย";
  });
  document.querySelectorAll("[data-example]").forEach(button => {
    button.addEventListener("click", () => {
      for (const [name, value] of Object.entries(examples[button.dataset.example])) {
        document.getElementById(name).value = value;
      }
      result.hidden = true;
      status.textContent = "ใส่ข้อมูลตัวอย่างแล้ว กดประเมินเพื่อดูผล";
    });
  });
  form.addEventListener("submit", event => {
    event.preventDefault();
    if (!configuration || !form.reportValidity()) return;
    const raw = Object.fromEntries(new FormData(form));
    const { score, prediction } = predictStroke(raw, configuration);
    document.getElementById("score").textContent = (score * 100).toFixed(1);
    document.getElementById("score-bar").style.width = `${score * 100}%`;
    document.getElementById("classification").textContent = prediction
      ? "โมเดลจัดอยู่ในกลุ่ม stroke" : "โมเดลจัดอยู่ในกลุ่ม non-stroke";
    document.getElementById("result-description").textContent = prediction
      ? "ข้อมูลที่กรอกได้คะแนนถึงจุดตัดที่เลือกจากชุด validation"
      : "ข้อมูลที่กรอกได้คะแนนต่ำกว่าจุดตัดที่เลือกจากชุด validation";
    result.dataset.positive = String(Boolean(prediction));
    result.hidden = false;
    status.textContent = "ประเมินสำเร็จ ข้อมูลที่กรอกไม่ได้ถูกส่งออกจากเบราว์เซอร์";
    result.scrollIntoView({ behavior: "smooth", block: "nearest" });
  });
  fetch(new URL("model.json", document.baseURI))
    .then(response => {
      if (!response.ok) throw new Error("Model could not be loaded");
      return response.json();
    })
    .then(model => {
      configuration = model;
      document.querySelectorAll("[data-feature]").forEach(field => {
        const required = model.required_raw_features.includes(field.dataset.feature);
        field.hidden = !required;
        field.querySelector("input, select").disabled = !required;
      });
      const names = { random_forest: "Random Forest", logistic_regression: "Logistic Regression" };
      document.getElementById("model-name").textContent = names[model.model_name];
      document.getElementById("threshold").textContent = model.threshold.toFixed(4);
      const metrics = model.test_metrics;
      if (metrics) {
        document.getElementById("test-recall").textContent = `${(metrics.recall * 100).toFixed(1)}%`;
        document.getElementById("test-precision").textContent = `${(metrics.precision * 100).toFixed(1)}%`;
        document.getElementById("test-ap").textContent = metrics.pr_auc.toFixed(3);
        document.getElementById("test-count").textContent = metrics.rows;
      }
      submit.disabled = false;
      document.querySelectorAll("[data-example]").forEach(button => { button.disabled = false; });
      status.textContent = "โมเดลพร้อมใช้งาน";
    })
    .catch(() => {
      status.textContent = "โหลดโมเดลไม่สำเร็จ กรุณารีเฟรชหน้าแล้วลองใหม่";
      status.setAttribute("role", "alert");
    });
}
