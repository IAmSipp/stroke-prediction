const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const { predictStroke } = require("../web/app.js");

const root = path.resolve(__dirname, "..");
const readJson = name => JSON.parse(fs.readFileSync(path.join(root, name), "utf8"));
const readCsv = name => {
  const lines = fs.readFileSync(path.join(root, name), "utf8").trim().split(/\r?\n/);
  const columns = lines.shift().split(",");
  return lines.map(line => Object.fromEntries(line.split(",").map((value, index) => [columns[index], value])));
};
const model = readJson("web/model.json");
const evaluation = readJson("models/evaluation_results.json");
const checksum = crypto.createHash("sha256").update(fs.readFileSync(path.join(root, "models/final_model_bundle.joblib"))).digest("hex");
assert.equal(model.model_sha256, checksum);
assert.equal(evaluation.model_sha256, checksum);
const testMetrics = evaluation.metrics.find(row => row.model === "selected_model" && row.split === "test");
assert.deepEqual(model.test_metrics, testMetrics);

const inputs = readCsv("data/processed/test.csv");
const expected = new Map(readCsv("models/test_predictions.csv").map(row => [row.id, row]));
assert.equal(inputs.length, expected.size);
let missingBmiCount = 0;
let maxError = 0;
for (const row of inputs) {
  const reference = expected.get(row.id);
  const actual = predictStroke(row, model);
  const error = Math.abs(actual.score - Number(reference.model_score));
  maxError = Math.max(maxError, error);
  assert.ok(error < 1e-10, `Score mismatch for ${row.id}: ${error}`);
  assert.equal(actual.prediction, Number(reference.prediction));
  const minimal = Object.fromEntries(model.required_raw_features.map(name => [name, row[name]]));
  assert.deepEqual(predictStroke(minimal, model), actual);
  if (row.bmi === "") {
    missingBmiCount++;
    assert.deepEqual(predictStroke({ ...row, bmi: null }, model), actual);
  }
}
assert.ok(missingBmiCount > 0);
const boundaryScore = predictStroke(inputs[0], model).score;
assert.equal(predictStroke(inputs[0], { ...model, threshold: boundaryScore }).prediction, 1);
console.log(`Passed: ${inputs.length} Python/browser predictions; ${missingBmiCount} missing BMI cases; max score error ${maxError}.`);
