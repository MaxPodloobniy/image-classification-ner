const form = document.getElementById("verify-form");
const imageInput = document.getElementById("image-input");
const fileLabel = document.getElementById("file-label");
const preview = document.getElementById("preview");
const submitBtn = document.getElementById("submit-btn");
const resultDiv = document.getElementById("result");

// ── image preview ─────────────────────────────────────────────
imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) return;
  fileLabel.textContent = file.name;
  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";
});

// ── form submit ───────────────────────────────────────────────
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  setLoading(true);
  resultDiv.innerHTML = "";
  resultDiv.classList.add("hidden");

  try {
    const data = new FormData(form);
    const res = await fetch("/verify", { method: "POST", body: data });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Server error");
    }

    const json = await res.json();
    renderResult(json);
  } catch (err) {
    resultDiv.innerHTML = `<div class="error-msg">⚠ ${err.message}</div>`;
    resultDiv.classList.remove("hidden");
  } finally {
    setLoading(false);
  }
});

// ── render helpers ────────────────────────────────────────────
function setLoading(on) {
  submitBtn.disabled = on;
  submitBtn.innerHTML = on
    ? `<span class="spinner"></span>Verifying…`
    : "Verify";
}

function renderResult(data) {
  const pct = Math.round(data.confidence * 100);
  const isTrue = data.verdict;
  const icon = isTrue ? "✅" : "❌";
  const label = isTrue ? "TRUE" : "FALSE";
  const cls = isTrue ? "true" : "false";

  const animalBadges =
    data.extracted_animals.length > 0
      ? data.extracted_animals.map((a) => `<span class="badge">${a}</span>`).join("")
      : `<span class="badge-none">none found</span>`;

  resultDiv.innerHTML = `
    <div class="verdict ${cls}">${icon} Statement is ${label}</div>

    <div class="meta-row">
      <span class="meta-label">Predicted class</span>
      <span class="meta-value">${data.predicted_class}</span>
    </div>

    <div class="meta-row" style="align-items:flex-start; flex-direction:column; gap:6px;">
      <span class="meta-label">Confidence — ${pct}%</span>
      <div class="confidence-bar-wrap" style="width:100%">
        <div class="confidence-bar" style="width:${pct}%"></div>
      </div>
    </div>

    <div class="meta-row" style="align-items:flex-start; flex-direction:column; gap:6px;">
      <span class="meta-label">Animals found in text</span>
      <div class="badges">${animalBadges}</div>
    </div>
  `;

  resultDiv.classList.remove("hidden");
}
