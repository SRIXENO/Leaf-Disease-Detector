/* script.js
   Smart Plant Doctor - frontend
   - voice default OFF
   - speaks only when user opens remedy and toggles voice ON
   - supports en/ta/hi selection for voice language where browser supports TTS
   - uses backend /predict endpoint, expects remedy_translated if the server handled translation
   - fetches OpenWeather client-side using window.ENV_OPENWEATHER_KEY (if provided)
*/

(() => {
  "use strict";

  // CONFIG
  const ENDPOINT = "/predict";
  const FETCH_TIMEOUT = 20000; // ms
  const WEATHER_TIMEOUT = 9000;
  const SPEECH_SETTINGS = {
    en: { lang: "en-IN", rate: 1.0, pitch: 1.0 },
    ta: { lang: "ta-IN", rate: 0.95, pitch: 1.0 },
    hi: { lang: "hi-IN", rate: 1.0, pitch: 1.0 },
  };

  // STATE
  const state = {
    voiceEnabled: false, // OFF by default
    userLocation: { lat: null, lon: null },
    isAnalyzing: false,
    lastResponse: null,
  };

  // DOM nodes
  const nodes = {
    form: document.getElementById("uploadForm"),
    fileInput: document.getElementById("fileInput"),
    langSelect: document.getElementById("langSelect"),
    result: document.getElementById("result"),
    previewWrap: document.getElementById("previewWrap"),
    imagePreview: document.getElementById("imagePreview"),
    weatherWidget: document.getElementById("weatherWidget"),
    analyzeBtn: document.getElementById("analyzeBtn"),
    resetBtn: document.getElementById("resetBtn"),
  };

  // Utility: timeout fetch
  function fetchWithTimeout(url, opts = {}, timeout = FETCH_TIMEOUT) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("Request timed out")), timeout);
      fetch(url, opts).then((res) => {
        clearTimeout(timer);
        resolve(res);
      }).catch((err) => {
        clearTimeout(timer);
        reject(err);
      });
    });
  }

  // initialize
  function init() {
    if (!nodes.form || !nodes.fileInput || !nodes.langSelect || !nodes.result) {
      console.error("Core DOM elements missing");
      return;
    }

    // prevent native form submission from reloading
    nodes.form.addEventListener("submit", onSubmit);
    nodes.resetBtn.addEventListener("click", resetUI);

    nodes.fileInput.addEventListener("change", onFileChange);

    // attempt geolocation (non-blocking)
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition((pos) => {
        state.userLocation.lat = pos.coords.latitude;
        state.userLocation.lon = pos.coords.longitude;
        // show quick weather if possible
        tryPopulateWeatherWidget();
      }, (err) => {
        console.warn("Location denied/failed:", err && err.message);
      }, { timeout: 7000 });
    }

    // hide preview initially
    if (nodes.previewWrap) nodes.previewWrap.style.display = "none";
  }

  // preview handler
  function onFileChange(e) {
    const file = e.target.files && e.target.files[0];
    if (!file) {
      nodes.previewWrap.style.display = "none";
      nodes.imagePreview.src = "";
      return;
    }
    // show preview
    const url = URL.createObjectURL(file);
    nodes.imagePreview.onload = () => URL.revokeObjectURL(url);
    nodes.imagePreview.src = url;
    nodes.previewWrap.style.display = "block";
  }

  // Reset UI
  function resetUI() {
    nodes.fileInput.value = "";
    nodes.previewWrap.style.display = "none";
    nodes.imagePreview.src = "";
    nodes.result.style.display = "none";
    nodes.result.innerHTML = "";
    state.lastResponse = null;
  }

  // Quick weather widget using ENV key if present
  async function tryPopulateWeatherWidget() {
    const key = window.ENV_OPENWEATHER_KEY || "";
    const wrap = nodes.weatherWidget;
    if (!key || !state.userLocation.lat) {
      wrap.innerHTML = `<div id="weatherContent">📍 Location: Unknown<br/>🌤️ Weather: N/A</div>`;
      return;
    }
    try {
      const out = await fetchWeather(state.userLocation.lat, state.userLocation.lon);
      wrap.innerHTML = `<div id="weatherContent">📍 ${escapeHtml(out.city)}<br/>🌤️ ${escapeHtml(out.condition)}, ${escapeHtml(out.temp)}°C<br/>⚠️ Risk: ${escapeHtml(out.risk)} (Humidity: ${escapeHtml(out.humidity)}%)</div>`;
    } catch (err) {
      wrap.innerHTML = `<div id="weatherContent">⚠️ Could not load weather</div>`;
    }
  }

  async function fetchWeather(lat, lon) {
    const key = window.ENV_OPENWEATHER_KEY || "";
    if (!key) throw new Error("OpenWeather key not provided to frontend");
    const url = `https://api.openweathermap.org/data/2.5/weather?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}&appid=${encodeURIComponent(key)}&units=metric`;
    const res = await fetchWithTimeout(url, {}, WEATHER_TIMEOUT);
    if (!res.ok) throw new Error("Weather fetch failed");
    const j = await res.json();
    const condition = (j.weather && j.weather[0] && j.weather[0].main) || "N/A";
    const temp = Math.round((j.main && j.main.temp) || 0);
    const humidity = (j.main && j.main.humidity) || "?";
    const city = j.name || "Unknown";
    let risk = "Low";
    if (humidity > 75 || condition === "Rain" || condition === "Thunderstorm") risk = "High";
    else if (humidity > 55 || condition === "Clouds" || condition === "Mist") risk = "Moderate";
    return { city, condition, temp, humidity, risk };
  }

  // form submit handler (main flow)
  async function onSubmit(e) {
    e.preventDefault();
    e.stopPropagation();

    if (state.isAnalyzing) return; // prevent double submit
    const file = nodes.fileInput.files && nodes.fileInput.files[0];
    if (!file) {
      alert("Please select an image file first.");
      return;
    }

    // show analyzing UI
    showAnalyzing();

    // prefetch weather client-side (optional) so we can show to user
    let clientWeather = { city: "Unknown", condition: "N/A", temp: "?", humidity: "?", risk: "N/A" };
    if (state.userLocation.lat) {
      try {
        clientWeather = await fetchWeather(state.userLocation.lat, state.userLocation.lon);
      } catch (err) {
        // ignore, we'll still call server
      }
    }

    // prepare form data
    const lang = nodes.langSelect.value || "en";
    const form = new FormData();
    form.append("file", file);

    // build url with optional lat/lon
    let url = `${ENDPOINT}?lang=${encodeURIComponent(lang)}`;
    if (state.userLocation.lat) {
      url += `&lat=${encodeURIComponent(state.userLocation.lat)}&lon=${encodeURIComponent(state.userLocation.lon)}`;
    }

    try {
      state.isAnalyzing = true;
      const res = await fetchWithTimeout(url, { method: "POST", body: form }, FETCH_TIMEOUT);
      let body;
      try { body = await res.json(); } catch (err) { throw new Error("Invalid JSON from server"); }
      if (!res.ok) {
        throw new Error(body && body.error ? body.error : `Server returned ${res.status}`);
      }
      state.lastResponse = body;
      renderResult(body, clientWeather);
    } catch (err) {
      console.error("Predict failed:", err);
      nodes.result.style.display = "block";
      nodes.result.innerHTML = `<div style="padding:12px;"><strong style="color:#c62828">Error:</strong> ${escapeHtml(err.message || String(err))}</div>`;
    } finally {
      state.isAnalyzing = false;
    }
  }

  // show analyzing UI
  function showAnalyzing() {
    nodes.result.style.display = "block";
    nodes.result.innerHTML = `<div style="padding:18px;text-align:center;"><div class="loader" aria-hidden="true"></div><div style="margin-top:8px;color:#666">Analyzing image... please wait</div></div>`;
  }

  // render results
  function renderResult(data, weather) {
    // normalization
    const pred = data.prediction || "Unknown";
    const conf = (typeof data.confidence === "number") ? data.confidence.toFixed(2) : (data.confidence || "?");
    const severity = data.severity || "Unknown";
    // Use pre-translated remedy if server provided, otherwise fallback to original remedy text.
    const remedyText = data.remedy_translated || data.remedy || "No remedy available";
    const gradcam = data.gradcam_image || "";
    const reportPdf = data.report_pdf || "";

    // build html
    const pdfHtml = reportPdf ? `<a class="pdf-btn" href="${escapeHtml(reportPdf)}" target="_blank">📄 Download Report</a>` : "";
    const gradcamHtml = gradcam ? `<div style="margin-top:10px;"><img src="${escapeHtml(gradcam)}" style="max-width:100%;border-radius:8px;box-shadow:0 6px 18px rgba(0,0,0,0.08)"></div>` : "";

    nodes.result.innerHTML = `
      <h2>🧠 AI Analysis Result</h2>
      <div style="display:flex;gap:14px;flex-wrap:wrap">
        <div style="flex:1;min-width:220px">
          <p><strong>Disease:</strong> ${escapeHtml(pred)}</p>
          <p><strong>Confidence:</strong> ${escapeHtml(conf)}%</p>
          <p><strong>Severity:</strong> ${escapeHtml(severity)}</p>
        </div>
        <div style="min-width:220px">
          <p><strong>📍 Location:</strong> ${escapeHtml(weather.city || (data.weather && data.weather.city) || "Unknown")}</p>
          <p><strong>🌤️ Weather:</strong> ${escapeHtml(weather.condition || (data.weather && data.weather.condition) || "N/A")}, ${escapeHtml(weather.temp)}°C</p>
          <p><strong>⚠️ Weather Risk:</strong> ${escapeHtml(weather.risk || (data.weather && data.weather.risk) || "N/A")} (Humidity: ${escapeHtml(weather.humidity || (data.weather && data.weather.humidity) || "?")}%)</p>
        </div>
      </div>

      <hr style="margin:12px 0;border:none;border-top:1px solid #eee">

      <div class="remedy-header" id="remedyHeader">
        <div>
          <strong>🌿 Remedy & Precautions</strong><br><small style="color:#666">Tap to expand — voice will read only when enabled</small>
        </div>
        <div style="display:flex;gap:8px;align-items:center">
          ${pdfHtml}
          <button id="copyRemedyBtn" class="copy-btn">📋 Copy</button>
          <span id="toggleRemedy" style="font-size:20px;cursor:pointer">⬇️</span>
        </div>
      </div>

      <div id="treatment" class="remedy-body" style="display:none">
        ${formatRemedyHtml(remedyText)}
        <div style="margin-top:10px;">
          <button id="voiceToggle" class="voice-toggle off">🔇 Voice: OFF</button>
        </div>
      </div>

      <div style="margin-top:12px">
        <h4 style="margin:0 0 8px 0">Other possibilities</h4>
        <ul id="otherPossibilities" style="padding-left:18px;color:#444">
          ${(Array.isArray(data.suggestions) && data.suggestions.length) ? data.suggestions.map(s => `<li>${escapeHtml(String(s[0]||"Unknown"))} — ${escapeHtml(String(s[1]||""))}%</li>`).join("") : "<li>None</li>"}
        </ul>
      </div>
      ${gradcamHtml}
    `;

    // attach interactions
    attachResultInteractions(remedyText, data);
  }

  // attach controls to rendered result
  function attachResultInteractions(remedyText, serverData) {
    const toggle = document.getElementById("toggleRemedy");
    const treat = document.getElementById("treatment");
    const copyBtn = document.getElementById("copyRemedyBtn");
    const voiceBtn = document.getElementById("voiceToggle");
    const header = document.getElementById("remedyHeader");

    // safety: remove duplicate listeners by cloning where needed
    if (toggle) replaceWithClone(toggle);
    if (copyBtn) replaceWithClone(copyBtn);
    if (voiceBtn) replaceWithClone(voiceBtn);
    if (header) replaceWithClone(header);

    // reselect after clone
    const toggle2 = document.getElementById("toggleRemedy");
    const copy2 = document.getElementById("copyRemedyBtn");
    const voice2 = document.getElementById("voiceToggle");
    const header2 = document.getElementById("remedyHeader");

    // toggle behavior
    if (toggle2) {
      toggle2.addEventListener("click", () => {
        const visible = treat.style.display === "block";
        treat.style.display = visible ? "none" : "block";
        toggle2.textContent = visible ? "⬇️" : "⬆️";
        // speak only if turning visible AND voice is ON
        if (!visible && state.voiceEnabled) {
          // choose voice language: use frontend lang selector to set speech lang
          const lang = (nodes.langSelect && nodes.langSelect.value) || "en";
          const text = (serverData.remedy_translated && serverData.remedy_translated.trim()) ? serverData.remedy_translated : serverData.remedy || "";
          speakText(text, lang);
        } else {
          cancelSpeech();
        }
      });
    }

    // header click toggles too (larger target)
    if (header2) {
      header2.addEventListener("click", () => {
        // click the small toggle to share behavior
        const t = document.getElementById("toggleRemedy");
        if (t) t.click();
      });
    }

    // copy button
    if (copy2) {
      copy2.addEventListener("click", async () => {
        const raw = (serverData.remedy_translated && serverData.remedy_translated.trim()) ? serverData.remedy_translated : serverData.remedy || "";
        if (!raw) return alert("No remedy text to copy.");
        try {
          await navigator.clipboard.writeText(raw);
          copy2.textContent = "✅ Copied";
          setTimeout(() => (copy2.textContent = "📋 Copy"), 1600);
        } catch (err) {
          alert("Copy failed — select and copy manually.");
        }
      });
    }

    // voice toggle inside result (starts OFF)
    if (voice2) {
      updateVoiceButtonUI(voice2, state.voiceEnabled);
      voice2.addEventListener("click", () => {
        state.voiceEnabled = !state.voiceEnabled;
        updateVoiceButtonUI(voice2, state.voiceEnabled);
        if (!state.voiceEnabled) cancelSpeech();
        else {
          // if remedy is visible and user turned on voice, speak immediately
          if (treat.style.display === "block") {
            const lang = (nodes.langSelect && nodes.langSelect.value) || "en";
            const text = (serverData.remedy_translated && serverData.remedy_translated.trim()) ? serverData.remedy_translated : serverData.remedy || "";
            speakText(text, lang);
          }
        }
      });
    }
  }

  function updateVoiceButtonUI(btn, enabled) {
    if (!btn) return;
    btn.textContent = enabled ? "🔈 Voice: ON" : "🔇 Voice: OFF";
    if (enabled) btn.classList.remove("off"); else btn.classList.add("off");
  }

  // speak cleaned text, using language selection to choose voice locale
  function speakText(rawText, langCode = "en") {
    if (!rawText || !state.voiceEnabled) return;
    const cleaned = cleanTextForSpeech(rawText);
    const settings = SPEECH_SETTINGS[langCode] || SPEECH_SETTINGS.en;
    try {
      // cancel any previous
      speechSynthesis.cancel();
    } catch (e) {}

    // choose best voice available for the lang prefix
    const utter = new SpeechSynthesisUtterance(cleaned);
    utter.lang = settings.lang;
    utter.rate = settings.rate;
    utter.pitch = settings.pitch;
    utter.onend = () => { /* optional feedback */ };
    utter.onerror = (e) => console.warn("TTS error", e);
    speechSynthesis.speak(utter);
  }

  function cancelSpeech() {
    try { speechSynthesis.cancel(); } catch (e) {}
  }

  // sanitize + prepare for speech (remove emojis/markdown but keep natural sentences)
  function cleanTextForSpeech(s) {
    if (!s) return "";
    let t = String(s);
    // If Tamil or Hindi, we assume backend provided native script (no transliteration)
    // remove markdown headers and bullet markers -> convert to sentences
    t = t.replace(/#+\s*/g, " ");
    t = t.replace(/\*\*(.*?)\*\*/g, "$1");
    t = t.replace(/[-•▪►➡–—•]/g, ". ");
    // remove common emojis (unicode ranges)
    t = t.replace(/[\u{1F300}-\u{1F6FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, " ");
    // collapse whitespace and convert newlines to periods
    t = t.replace(/(\r\n|\n|\r)/g, ". ");
    t = t.replace(/\s{2,}/g, " ");
    t = t.trim();
    if (!/[.!?]$/.test(t)) t = t + ".";
    return t;
  }

  // Helpers: escape HTML
  function escapeHtml(unsafe) {
    if (unsafe === undefined || unsafe === null) return "";
    return String(unsafe).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#039;");
  }

  // replace element with clone to remove previous listeners
  function replaceWithClone(el) {
    if (!el || !el.parentNode) return;
    const clone = el.cloneNode(true);
    el.parentNode.replaceChild(clone, el);
  }

  // format remedy for HTML block (keeps some bullets and bold)
  function formatRemedyHtml(text) {
    if (!text) return "<em>No remedy provided</em>";
    let out = String(text);
    // headings (###) -> emoji + bold
    out = out.replace(/###\s*(.+)/g, "🌿 <strong>$1</strong>");
    out = out.replace(/##\s*(.+)/g, "🌾 <strong>$1</strong>");
    out = out.replace(/#\s*(.+)/g, "🌱 <strong>$1</strong>");
    // bold
    out = out.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    // bullets -> emoji bullet
    out = out.replace(/(^|\n)[\-\*]\s+/g, "$1🍀 ");
    // newlines
    out = out.replace(/\r\n/g, "\n");
    out = out.replace(/\n{2,}/g, "<br><br>");
    out = out.replace(/\n/g, "<br>");
    return out;
  }

  // bootstrap
  document.addEventListener("DOMContentLoaded", () => {
    init();
    // update weather widget if possible
    tryPopulateWeatherWidget();
    // log default state for debugging
    console.log("Smart Plant Doctor frontend initialized. Voice default OFF.");
  });

  window.__SPD = {
    state,
    speakText,
    fetchWeather,
    cleanTextForSpeech
  };

})();
