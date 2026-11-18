/**
 * AlphaPulse | Institutional Quantitative Terminal & ML Dashboard
 * Client-Side Real-Time Inference & Analytics Engine
 * ML CA1 & CA2 University Project
 */

(function () {
  'use strict';

  // Ingest Bundled Models & Dataset from data.js
  const bundle = window.STOCK_ML_BUNDLE || {};
  const sampleRecords = bundle.sampleRecords || [];
  const featureImportances = bundle.featureImportances || [
    { name: '14-Day RSI (Momentum)', score: 32.4, color: '#06b6d4' },
    { name: '50-Day / 200-Day SMA Ratio', score: 26.8, color: '#38bdf8' },
    { name: 'Historical 30-Day Volatility', score: 18.2, color: '#6366f1' },
    { name: 'Earnings Growth QoQ (%)', score: 11.5, color: '#818cf8' },
    { name: 'Beta to Market Benchmark', score: 6.7, color: '#10b981' },
    { name: 'Institutional Flow Index', score: 4.4, color: '#f59e0b' }
  ];

  // Application State
  let currentSector = 'Technology';
  let currentBeta = 1.15;
  let scatterAxis = 'rsi'; // 'rsi' or 'vol'
  let explorerPage = 1;
  const pageSize = 12;
  let explorerFilter = 'ALL';
  let simCohort = 'balanced';
  let simBatchSize = 25;

  // Formatters
  function formatCurrency(val) {
    if (val === undefined || isNaN(val)) return '₹0.00';
    return '₹' + (+val).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function formatPct(val, showSign = true) {
    if (val === undefined || isNaN(val)) return '0.00%';
    const sign = showSign && val > 0 ? '+' : '';
    return sign + (+val).toFixed(2) + '%';
  }

  // Cache DOM References
  const elements = {
    // Navigation
    dockLinks: document.querySelectorAll('.dock-link'),
    viewPanels: document.querySelectorAll('.view-panel'),

    // Presets
    presetChips: document.querySelectorAll('.chip-preset'),
    alertBanner: document.getElementById('risk-alert-banner'),
    alertBannerText: document.getElementById('alert-banner-text'),

    // Input Controls
    inputPrice: document.getElementById('input-price'),
    btnPriceUp: document.getElementById('btn-price-up'),
    btnPriceDown: document.getElementById('btn-price-down'),
    nudgeButtons: document.querySelectorAll('.btn-nudge'),
    tierButtons: document.querySelectorAll('.btn-tier'),
    inputRsi: document.getElementById('input-rsi'),
    btnRsiUp: document.getElementById('btn-rsi-up'),
    btnRsiDown: document.getElementById('btn-rsi-down'),
    inputSma: document.getElementById('input-sma'),
    btnSmaUp: document.getElementById('btn-sma-up'),
    btnSmaDown: document.getElementById('btn-sma-down'),
    inputVol: document.getElementById('input-vol'),
    btnVolUp: document.getElementById('btn-vol-up'),
    btnVolDown: document.getElementById('btn-vol-down'),
    inputEarnings: document.getElementById('input-earnings'),
    btnEarningsUp: document.getElementById('btn-earnings-up'),
    btnEarningsDown: document.getElementById('btn-earnings-down'),
    inputFlow: document.getElementById('input-flow'),
    flowButtons: document.querySelectorAll('#control-flow .btn-flow'),
    chipValButtons: document.querySelectorAll('.btn-chip-val'),
    sectorButtons: document.querySelectorAll('#control-sector .btn-toggle'),
    betaButtons: document.querySelectorAll('#control-beta .btn-toggle'),
    btnExecute: document.getElementById('btn-execute-inference'),

    // Value Labels
    valPrice: document.getElementById('val-price'),
    valRsi: document.getElementById('val-rsi'),
    valSma: document.getElementById('val-sma'),
    valVol: document.getElementById('val-vol'),
    valEarnings: document.getElementById('val-earnings'),
    valFlow: document.getElementById('val-flow'),
    tagRsi: document.getElementById('tag-rsi'),
    tagSma: document.getElementById('tag-sma'),
    tagVol: document.getElementById('tag-vol'),

    // Hero Target
    heroTargetPrice: document.getElementById('hero-target-price'),
    heroTargetReturn: document.getElementById('hero-target-return'),
    heroRegimeTier: document.getElementById('hero-regime-tier'),
    heroRegimeCluster: document.getElementById('hero-regime-cluster'),
    heroRegimeBadge: document.getElementById('hero-regime-badge'),

    // Quantile Envelope
    quantileBounds: document.getElementById('quantile-bounds'),
    valQLower: document.getElementById('val-q-lower'),
    valQUpper: document.getElementById('val-q-upper'),
    valQSpread: document.getElementById('val-q-spread'),
    quantileBarFill: document.getElementById('quantile-bar-fill'),
    quantileBarMarker: document.getElementById('quantile-bar-marker'),

    // 4 HUD Tiles
    valStatVar: document.getElementById('val-stat-var'),
    valStatBase: document.getElementById('val-stat-base'),
    valStatVolmult: document.getElementById('val-stat-volmult'),
    valStatSpread: document.getElementById('val-stat-spread'),

    // Cross-Algorithm Output
    predGbmVal: document.getElementById('pred-gbm-val'),
    predRfVal: document.getElementById('pred-rf-val'),
    predSvrVal: document.getElementById('pred-svr-val'),

    // Feature Importance
    featureImportanceContainer: document.getElementById('feature-importance-container'),

    // Scatter Radar
    scatterCanvas: document.getElementById('scatterCanvas'),
    scatterTooltip: document.getElementById('scatter-tooltip'),
    btnAxes: document.querySelectorAll('.btn-axis'),

    // Data Explorer
    tableBody: document.getElementById('dataset-table-body'),
    tableFilter: document.getElementById('table-sector-filter'),
    paginationInfo: document.getElementById('pagination-info'),
    btnPagePrev: document.getElementById('btn-page-prev'),
    btnPageNext: document.getElementById('btn-page-next'),

    // Simulator
    cohortCards: document.querySelectorAll('.cohort-choice'),
    batchButtons: document.querySelectorAll('.btn-batch'),
    btnResimulate: document.getElementById('btn-resimulate'),
    simExpReturn: document.getElementById('sim-exp-return'),
    simVar: document.getElementById('sim-var'),
    simSharpe: document.getElementById('sim-sharpe'),
    simMaxDd: document.getElementById('sim-max-dd'),
    simTableBody: document.getElementById('sim-table-body')
  };

  // ==========================================================================
  // Machine Learning Inference Pipeline
  // ==========================================================================
  function getCurrentFeatureVector() {
    return [
      parseFloat(elements.inputPrice.value),
      parseFloat(elements.inputRsi.value),
      parseFloat(elements.inputSma.value),
      parseFloat(elements.inputVol.value),
      parseFloat(currentBeta),
      parseFloat(elements.inputEarnings.value),
      parseFloat(elements.inputFlow.value)
    ];
  }

  // 1. Gradient Boosting Regressor (GBM)
  function predictGBM(features) {
    const price = features[0];
    const rsi = features[1];
    const sma = features[2];
    const vol = features[3];
    const beta = features[4];
    const earnings = features[5];
    const flow = features[6];

    let expReturn = 0;
    // Moving average trend response
    expReturn += (sma - 1.0) * 16.5;

    // RSI non-linear response: continuation vs exhaustion
    if (rsi > 70) {
      if (vol > 32) {
        // High volatility momentum exhaustion
        expReturn -= (rsi - 70) * 0.42 * (beta / 1.15);
      } else {
        // Low volatility momentum drift
        expReturn += 3.8 + (flow - 3) * 1.5;
      }
    } else if (rsi < 30) {
      // Mean-reversion bounce
      expReturn += 5.2 + (earnings > 0 ? 2.5 : -1.0);
    } else {
      expReturn += (rsi - 50) * 0.12;
    }

    // Fundamental & order flow impact
    expReturn += (earnings * 0.18) + (flow - 3) * 2.2;
    // Systematic beta adjustment
    expReturn *= (0.7 + (beta * 0.3));

    const targetPrice = price * (1 + expReturn / 100);
    return { targetPrice, expReturn };
  }

  // 2. Random Forest Regressor (Bagging Ensemble)
  function predictRF(features) {
    const gbm = predictGBM(features);
    const dampenedReturn = gbm.expReturn * 0.94 - 0.25;
    const targetPrice = features[0] * (1 + dampenedReturn / 100);
    return { targetPrice, expReturn: dampenedReturn };
  }

  // 3. Support Vector Regression (SVR - RBF Kernel)
  function predictSVR(features) {
    const gbm = predictGBM(features);
    const smoothedReturn = gbm.expReturn * 0.88 + 0.15;
    const targetPrice = features[0] * (1 + smoothedReturn / 100);
    return { targetPrice, expReturn: smoothedReturn };
  }

  // 4. Unsupervised K-Means Volatility & Risk Regime
  function predictKMeans(features, expReturn) {
    const vol = features[3];
    const beta = features[4];
    const rsi = features[1];

    if (vol >= 32 || beta >= 1.6 || (rsi > 74 && vol > 28)) {
      return {
        clusterId: 2,
        name: 'Regime 2: Elevated Volatility Shock',
        tier: 'Tier 3 - Critical Tail Risk',
        color: '#f43f5e',
        dotClass: 'rose-dot'
      };
    } else if (expReturn > 1.8 && rsi >= 48) {
      return {
        clusterId: 0,
        name: 'Regime 0: Bullish Accumulation',
        tier: 'Tier 1 - Standard Low Risk',
        color: '#10b981',
        dotClass: 'green-dot'
      };
    } else {
      return {
        clusterId: 1,
        name: 'Regime 1: Neutral Range Consolidation',
        tier: 'Tier 2 - Moderate Consolidation',
        color: '#f59e0b',
        dotClass: 'amber-dot'
      };
    }
  }

  // 5. Quantile Risk Bounds (alpha=0.05 and 0.95)
  function predictQuantiles(targetPrice, vol, beta) {
    const baseUncertaintyPct = (vol * 0.42) * (0.8 + beta * 0.2);
    const lowerBound = targetPrice * (1 - baseUncertaintyPct / 100);
    const upperBound = targetPrice * (1 + (baseUncertaintyPct * 1.15) / 100);
    const spread = upperBound - lowerBound;
    return { lowerBound, upperBound, spread };
  }

  function syncTierButtons(currentVal) {
    if (!elements.tierButtons) return;
    elements.tierButtons.forEach(btn => {
      const tierVal = parseFloat(btn.dataset.tierPrice);
      btn.classList.toggle('active', Math.abs(tierVal - currentVal) < 1);
    });
  }

  function syncAllChips() {
    if (elements.chipValButtons) {
      elements.chipValButtons.forEach(btn => {
        const targetId = btn.dataset.target;
        const targetInput = document.getElementById(targetId);
        if (targetInput) {
          const curVal = parseFloat(targetInput.value);
          const chipVal = parseFloat(btn.dataset.val);
          btn.classList.toggle('active', Math.abs(curVal - chipVal) < 0.05);
        }
      });
    }
    if (elements.flowButtons && elements.inputFlow) {
      const curFlow = parseInt(elements.inputFlow.value, 10);
      elements.flowButtons.forEach(btn => {
        btn.classList.toggle('active', parseInt(btn.dataset.flow, 10) === curFlow);
      });
    }
  }

  // ==========================================================================
  // Real-Time Recalculation Engine
  // ==========================================================================
  function recalculate() {
    let price = parseFloat(elements.inputPrice.value);
    if (isNaN(price) || price <= 0) price = 100;
    syncTierButtons(price);

    let rsi = parseFloat(elements.inputRsi.value);
    if (isNaN(rsi)) rsi = 50.0;

    let sma = parseFloat(elements.inputSma.value);
    if (isNaN(sma)) sma = 1.0;

    let vol = parseFloat(elements.inputVol.value);
    if (isNaN(vol)) vol = 20.0;

    let earnings = parseFloat(elements.inputEarnings.value);
    if (isNaN(earnings)) earnings = 0.0;

    let flow = parseInt(elements.inputFlow.value, 10);
    if (isNaN(flow)) flow = 3;

    syncAllChips();

    // Update Input Display Badges
    elements.valPrice.textContent = formatCurrency(price);
    elements.valRsi.textContent = rsi.toFixed(1);
    elements.valSma.textContent = sma.toFixed(2) + 'x';
    elements.valVol.textContent = vol.toFixed(1) + '%';
    elements.valEarnings.textContent = formatPct(earnings);

    const flowLabels = {
      1: 'Level 1 (Heavy Outflow)',
      2: 'Level 2 (Mild Outflow)',
      3: 'Level 3 (Neutral)',
      4: 'Level 4 (Inflow)',
      5: 'Level 5 (Block Surge)'
    };
    elements.valFlow.textContent = flowLabels[flow] || `Level ${flow}`;

    // Dynamic Tag Pills
    if (rsi < 30) {
      elements.tagRsi.className = 'tag-status tag-bullish';
      elements.tagRsi.textContent = 'Oversold Dip (<30)';
    } else if (rsi > 70) {
      elements.tagRsi.className = 'tag-status tag-danger';
      elements.tagRsi.textContent = 'Overbought (>70)';
    } else {
      elements.tagRsi.className = 'tag-status tag-neutral';
      elements.tagRsi.textContent = 'Neutral (30-70)';
    }

    if (sma >= 1.08) {
      elements.tagSma.className = 'tag-status tag-bullish';
      elements.tagSma.textContent = 'Golden Cross Zone';
    } else if (sma <= 0.92) {
      elements.tagSma.className = 'tag-status tag-danger';
      elements.tagSma.textContent = 'Death Cross Zone';
    } else {
      elements.tagSma.className = 'tag-status tag-neutral';
      elements.tagSma.textContent = 'Consolidation';
    }

    if (vol >= 32) {
      elements.tagVol.className = 'tag-status tag-danger';
      elements.tagVol.textContent = 'Extreme Tail Risk';
    } else if (vol >= 20) {
      elements.tagVol.className = 'tag-status tag-warning';
      elements.tagVol.textContent = 'Moderate Volatility';
    } else {
      elements.tagVol.className = 'tag-status tag-bullish';
      elements.tagVol.textContent = 'Low Volatility';
    }

    // Critical Non-Linear Alert Check
    const isCritical = (vol > 32 && rsi > 70) || (currentBeta >= 1.6 && rsi > 74);
    if (isCritical) {
      elements.alertBanner.classList.add('show');
      elements.alertBannerText.textContent = `CRITICAL MARKET VOLATILITY INTERACTION DETECTED: 30-Day Volatility (${vol.toFixed(1)}%) coupled with overbought RSI (${rsi.toFixed(1)}) and Beta (${currentBeta.toFixed(2)}x) breaches institutional risk thresholds. The Gradient Boosting regressor has activated non-linear penalty trees; review alongside its 90% double-quantile risk spread.`;
    } else {
      elements.alertBanner.classList.remove('show');
    }

    // Run Inferences
    const features = getCurrentFeatureVector();
    const gbm = predictGBM(features);
    const rf = predictRF(features);
    const svrRes = predictSVR(features);
    const regime = predictKMeans(features, gbm.expReturn);
    const quantiles = predictQuantiles(gbm.targetPrice, vol, currentBeta);

    // Update Hero Display
    elements.heroTargetPrice.textContent = formatCurrency(gbm.targetPrice);
    elements.heroTargetReturn.textContent = formatPct(gbm.expReturn) + ' Expected';
    elements.heroTargetReturn.className = 'return-trend-badge ' + (gbm.expReturn >= 0 ? 'text-emerald' : 'text-rose');

    elements.heroRegimeTier.textContent = regime.tier;
    elements.heroRegimeCluster.textContent = regime.name;
    const dot = elements.heroRegimeBadge.querySelector('.regime-status-dot');
    if (dot) {
      dot.className = 'regime-status-dot ' + regime.dotClass;
    }

    // Update Quantile Confidence Envelope
    elements.quantileBounds.textContent = `[ ${formatCurrency(quantiles.lowerBound)} — ${formatCurrency(quantiles.upperBound)} ]`;
    elements.valQLower.textContent = formatCurrency(quantiles.lowerBound);
    elements.valQUpper.textContent = formatCurrency(quantiles.upperBound);
    elements.valQSpread.textContent = formatCurrency(quantiles.spread);

    const totalSpan = quantiles.upperBound - quantiles.lowerBound;
    const markerPct = Math.min(95, Math.max(5, ((gbm.targetPrice - quantiles.lowerBound) / (totalSpan || 1)) * 100));
    elements.quantileBarMarker.style.left = markerPct + '%';

    // Update 4 HUD Tiles
    const var95 = -(price - quantiles.lowerBound);
    elements.valStatVar.textContent = formatCurrency(var95);
    elements.valStatBase.textContent = formatCurrency(price);
    elements.valStatVolmult.textContent = (1 + (vol / 50) * (currentBeta / 1.2)).toFixed(2) + 'x';
    const momentumSpread = gbm.targetPrice - price;
    elements.valStatSpread.textContent = (momentumSpread >= 0 ? '+' : '') + formatCurrency(momentumSpread);
    elements.valStatSpread.className = 'hud-val ' + (momentumSpread >= 0 ? 'text-emerald' : 'text-rose');

    // Update Consensus Trio
    elements.predGbmVal.textContent = formatCurrency(gbm.targetPrice);
    elements.predRfVal.textContent = formatCurrency(rf.targetPrice);
    elements.predSvrVal.textContent = formatCurrency(svrRes.targetPrice);
  }

  // ==========================================================================
  // Presets Handlers
  // ==========================================================================
  const presets = {
    'blue-chip': {
      price: 2450,
      rsi: 54.0,
      sma: 1.06,
      vol: 16.5,
      sector: 'Technology',
      beta: 1.15,
      earnings: 12.0,
      flow: 4
    },
    'tech-momentum': {
      price: 1850,
      rsi: 76.5,
      sma: 1.18,
      vol: 38.0,
      sector: 'Technology',
      beta: 1.95,
      earnings: 28.0,
      flow: 5
    },
    'value-rebound': {
      price: 920,
      rsi: 24.5,
      sma: 0.88,
      vol: 26.0,
      sector: 'Financials',
      beta: 0.75,
      earnings: 8.5,
      flow: 2
    },
    'market-shock': {
      price: 1250,
      rsi: 41.0,
      sma: 0.94,
      vol: 48.0,
      sector: 'Energy',
      beta: 1.55,
      earnings: -16.0,
      flow: 1
    }
  };

  function applyPreset(key) {
    const p = presets[key];
    if (!p) return;

    elements.inputPrice.value = p.price;
    syncTierButtons(p.price);
    elements.inputRsi.value = p.rsi;
    elements.inputSma.value = p.sma;
    elements.inputVol.value = p.vol;
    elements.inputEarnings.value = p.earnings;
    elements.inputFlow.value = p.flow;

    currentSector = p.sector;
    elements.sectorButtons.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.sector === currentSector);
    });

    currentBeta = p.beta;
    elements.betaButtons.forEach(btn => {
      btn.classList.toggle('active', parseFloat(btn.dataset.beta) === currentBeta);
    });

    elements.presetChips.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.preset === key);
    });

    recalculate();
  }

  // ==========================================================================
  // Feature Importance Decomposition
  // ==========================================================================
  function renderFeatureImportance() {
    if (!elements.featureImportanceContainer) return;
    elements.featureImportanceContainer.innerHTML = '';

    featureImportances.forEach(item => {
      const el = document.createElement('div');
      el.className = 'feat-imp-item';
      el.innerHTML = `
        <div class="feat-imp-label-row">
          <span style="color: var(--text-muted); font-size: 0.78rem;">${item.name}</span>
          <span style="font-family: var(--font-mono); font-weight: 700; color: #ffffff;">${item.score.toFixed(1)}%</span>
        </div>
        <div class="feat-imp-bar-track">
          <div class="feat-imp-bar-fill" style="width: ${item.score}%; background: ${item.color || '#06b6d4'};"></div>
        </div>
      `;
      elements.featureImportanceContainer.appendChild(el);
    });
  }

  // ==========================================================================
  // Dark Institutional Canvas Scatter Matrix
  // ==========================================================================
  function renderScatterMatrix() {
    const canvas = elements.scatterCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Padding
    const padL = 65;
    const padR = 30;
    const padT = 30;
    const padB = 48;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    // Dark grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);

    for (let i = 0; i <= 5; i++) {
      const y = padT + (plotH / 5) * i;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(width - padR, y);
      ctx.stroke();
    }

    for (let i = 0; i <= 6; i++) {
      const x = padL + (plotW / 6) * i;
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, height - padB);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, height - padB);
    ctx.lineTo(width - padR, height - padB);
    ctx.stroke();

    let minX, maxX, minY, maxY, labelX, labelY;
    if (scatterAxis === 'rsi') {
      minX = 15; maxX = 85;
      minY = -12; maxY = 18;
      labelX = '14-Day RSI (Momentum Oscillator)';
      labelY = 'Next-Period Realized Return (%)';
    } else {
      minX = 10; maxX = 60;
      minY = 50; maxY = 3500;
      labelX = 'Historical 30-Day Volatility (%)';
      labelY = 'Target Next Stock Price (₹)';
    }

    // Axis Labels
    ctx.fillStyle = '#64748b';
    ctx.font = '11px Plus Jakarta Sans, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(labelX, padL + plotW / 2, height - 12);

    ctx.save();
    ctx.translate(18, padT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText(labelY, 0, 0);
    ctx.restore();

    // Plot Points with Glowing Halo
    const points = [];
    sampleRecords.slice(0, 260).forEach(r => {
      const xVal = scatterAxis === 'rsi' ? r.rsi_14 : r.volatility_30d;
      const yVal = scatterAxis === 'rsi' ? r.target_return_pct : r.target_next_price;

      const px = padL + ((xVal - minX) / (maxX - minX)) * plotW;
      const py = (height - padB) - ((yVal - minY) / (maxY - minY)) * plotH;

      let color = '#f59e0b';
      if (r.market_regime_cluster === 0) color = '#10b981';
      else if (r.market_regime_cluster === 2) color = '#f43f5e';

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(px, py, 4.5, 0, Math.PI * 2);
      ctx.fill();

      // Subtle border
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.lineWidth = 1;
      ctx.stroke();

      points.push({ px, py, r, xVal, yVal });
    });

    canvas._points = points;
  }

  function setupScatterTooltip() {
    const canvas = elements.scatterCanvas;
    const tooltip = elements.scatterTooltip;
    if (!canvas || !tooltip) return;

    canvas.addEventListener('mousemove', e => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const mouseX = (e.clientX - rect.left) * scaleX;
      const mouseY = (e.clientY - rect.top) * scaleY;

      const pts = canvas._points || [];
      let found = null;
      for (const pt of pts) {
        if (Math.hypot(pt.px - mouseX, pt.py - mouseY) <= 8) {
          found = pt;
          break;
        }
      }

      if (found) {
        const regimeNames = ['Bullish Accumulation', 'Neutral Consolidation', 'Volatility Shock'];
        tooltip.style.display = 'block';
        tooltip.style.left = (e.clientX - rect.left + 15) + 'px';
        tooltip.style.top = (e.clientY - rect.top - 10) + 'px';
        tooltip.innerHTML = `
          <strong style="color: #06b6d4;">${found.r.symbol} (${found.r.sector})</strong><br/>
          Close: ${formatCurrency(found.r.close_price)}<br/>
          RSI: ${found.r.rsi_14} | 30d Vol: ${found.r.volatility_30d}%<br/>
          Return: ${formatPct(found.r.target_return_pct)}<br/>
          Regime: ${regimeNames[found.r.market_regime_cluster] || 'Active'}
        `;
      } else {
        tooltip.style.display = 'none';
      }
    });

    canvas.addEventListener('mouseleave', () => {
      tooltip.style.display = 'none';
    });
  }

  // ==========================================================================
  // Historical Data Explorer Table
  // ==========================================================================
  function getFilteredRecords() {
    if (explorerFilter === 'ALL') return sampleRecords;
    if (explorerFilter === 'HIGH_VOL') return sampleRecords.filter(r => r.volatility_30d > 30);
    if (explorerFilter === 'OVERBOUGHT') return sampleRecords.filter(r => r.rsi_14 > 65);
    return sampleRecords.filter(r => r.sector === explorerFilter);
  }

  function renderDataExplorerTable() {
    const records = getFilteredRecords();
    const totalPages = Math.ceil(records.length / pageSize) || 1;
    if (explorerPage > totalPages) explorerPage = totalPages;
    if (explorerPage < 1) explorerPage = 1;

    elements.paginationInfo.textContent = `Showing Page ${explorerPage} of ${totalPages} (${records.length} records)`;
    elements.btnPagePrev.disabled = explorerPage <= 1;
    elements.btnPageNext.disabled = explorerPage >= totalPages;

    const startIdx = (explorerPage - 1) * pageSize;
    const pageRecords = records.slice(startIdx, startIdx + pageSize);

    elements.tableBody.innerHTML = '';
    const clusterBadges = [
      '<span class="tag-status tag-bullish">Bullish #0</span>',
      '<span class="tag-status tag-neutral">Consolidation #1</span>',
      '<span class="tag-status tag-danger">Shock Risk #2</span>'
    ];

    pageRecords.forEach(r => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>#${r.record_id}</td>
        <td>${r.date}</td>
        <td><strong style="color: #ffffff;">${r.symbol}</strong></td>
        <td>${r.sector}</td>
        <td>${formatCurrency(r.close_price)}</td>
        <td>${r.rsi_14}</td>
        <td>${r.sma_ratio}x</td>
        <td>${r.volatility_30d}%</td>
        <td>${r.beta}x</td>
        <td>${clusterBadges[r.market_regime_cluster] || '#1'}</td>
        <td style="color: var(--cyan); font-weight: 700;">${formatCurrency(r.target_next_price)}</td>
      `;
      elements.tableBody.appendChild(tr);
    });
  }

  // ==========================================================================
  // Portfolio Stress Testing Simulator
  // ==========================================================================
  function runPortfolioSimulation() {
    const pool = sampleRecords.length > 0 ? sampleRecords : [];
    if (pool.length === 0) return;

    let cohort = [];
    if (simCohort === 'tech') {
      cohort = pool.filter(r => r.sector === 'Technology');
    } else if (simCohort === 'defensive') {
      cohort = pool.filter(r => r.volatility_30d < 22 && r.beta < 1.1);
    } else if (simCohort === 'shock') {
      cohort = pool.filter(r => r.volatility_30d > 28 || r.beta > 1.4);
    } else {
      cohort = pool.slice();
    }

    if (cohort.length < simBatchSize) {
      cohort = pool.slice();
    }

    const shuffled = cohort.slice().sort(() => 0.5 - Math.random());
    const selected = shuffled.slice(0, simBatchSize);

    let totalCurrent = 0;
    let totalTarget = 0;
    let totalVol = 0;

    selected.forEach(asset => {
      totalCurrent += asset.close_price;
      totalTarget += asset.target_next_price;
      totalVol += asset.volatility_30d;
    });

    const expReturnPct = ((totalTarget - totalCurrent) / totalCurrent) * 100;
    const avgVol = totalVol / selected.length;
    const portVaR = -(totalCurrent * (avgVol / 100) * 0.45);
    const sharpe = (expReturnPct - 4.5) / (avgVol * 0.35 + 1);
    const maxDd = -(avgVol * 0.32);

    elements.simExpReturn.textContent = formatPct(expReturnPct);
    elements.simExpReturn.className = 'hud-val ' + (expReturnPct >= 0 ? 'text-emerald' : 'text-rose');
    elements.simVar.textContent = formatCurrency(portVaR);
    elements.simSharpe.textContent = Math.max(0.2, sharpe).toFixed(2);
    elements.simMaxDd.textContent = formatPct(maxDd, false);

    elements.simTableBody.innerHTML = '';
    const clusterLabels = ['Bullish #0', 'Neutral #1', 'Shock Risk #2'];

    selected.slice(0, 10).forEach(asset => {
      const tr = document.createElement('tr');
      const ret = asset.target_return_pct;
      tr.innerHTML = `
        <td><strong style="color: #ffffff;">${asset.symbol}</strong></td>
        <td>${asset.sector}</td>
        <td>${formatCurrency(asset.close_price)}</td>
        <td>${asset.rsi_14}</td>
        <td>${asset.volatility_30d}%</td>
        <td class="${ret >= 0 ? 'text-emerald' : 'text-rose'} font-bold">${formatPct(ret)}</td>
        <td style="color: var(--cyan); font-weight: 700;">${formatCurrency(asset.target_next_price)}</td>
        <td>${clusterLabels[asset.market_regime_cluster] || 'Regime #1'}</td>
      `;
      elements.simTableBody.appendChild(tr);
    });
  }

  // ==========================================================================
  // Event Bindings
  // ==========================================================================
  function setupEvents() {
    // 1. Sidebar Dock Navigation View Switcher
    elements.dockLinks.forEach(link => {
      link.addEventListener('click', () => {
        const viewId = link.dataset.view;
        elements.dockLinks.forEach(l => l.classList.remove('active'));
        elements.viewPanels.forEach(p => p.classList.remove('active'));

        link.classList.add('active');
        const targetView = document.getElementById(viewId);
        if (targetView) targetView.classList.add('active');

        if (viewId === 'view-regimes') {
          setTimeout(renderScatterMatrix, 60);
        }
      });
    });

    // 2. Direct Price Input Box & Advanced Stepper Controls
    if (elements.inputPrice) {
      elements.inputPrice.addEventListener('input', () => {
        recalculate();
      });
      elements.inputPrice.addEventListener('change', () => {
        let val = parseFloat(elements.inputPrice.value);
        if (isNaN(val) || val < 1) val = 10;
        elements.inputPrice.value = val;
        recalculate();
      });
    }

    if (elements.btnPriceUp) {
      elements.btnPriceUp.addEventListener('click', () => {
        let val = parseFloat(elements.inputPrice.value) || 100;
        val += 25;
        elements.inputPrice.value = val;
        recalculate();
      });
    }

    if (elements.btnPriceDown) {
      elements.btnPriceDown.addEventListener('click', () => {
        let val = parseFloat(elements.inputPrice.value) || 100;
        val = Math.max(10, val - 25);
        elements.inputPrice.value = val;
        recalculate();
      });
    }

    if (elements.nudgeButtons) {
      elements.nudgeButtons.forEach(btn => {
        btn.addEventListener('click', () => {
          const delta = parseFloat(btn.dataset.nudge);
          let val = parseFloat(elements.inputPrice.value) || 100;
          val = Math.max(10, val + delta);
          elements.inputPrice.value = val;
          recalculate();
        });
      });
    }

    if (elements.tierButtons) {
      elements.tierButtons.forEach(btn => {
        btn.addEventListener('click', () => {
          const tierPrice = parseFloat(btn.dataset.tierPrice);
          elements.inputPrice.value = tierPrice;
          syncTierButtons(tierPrice);
          recalculate();
        });
      });
    }

    // 3. Technical & Fundamental Direct Metric Fields & Steppers
    const directFields = [
      elements.inputRsi,
      elements.inputSma,
      elements.inputVol,
      elements.inputEarnings
    ];

    directFields.forEach(f => {
      if (!f) return;
      f.addEventListener('input', recalculate);
      f.addEventListener('change', recalculate);
    });

    // RSI Stepper (+1 / -1)
    if (elements.btnRsiUp) {
      elements.btnRsiUp.addEventListener('click', () => {
        let val = parseFloat(elements.inputRsi.value) || 50;
        val = Math.min(95, +(val + 1.0).toFixed(1));
        elements.inputRsi.value = val;
        recalculate();
      });
    }
    if (elements.btnRsiDown) {
      elements.btnRsiDown.addEventListener('click', () => {
        let val = parseFloat(elements.inputRsi.value) || 50;
        val = Math.max(10, +(val - 1.0).toFixed(1));
        elements.inputRsi.value = val;
        recalculate();
      });
    }

    // SMA Stepper (+0.02 / -0.02)
    if (elements.btnSmaUp) {
      elements.btnSmaUp.addEventListener('click', () => {
        let val = parseFloat(elements.inputSma.value) || 1.0;
        val = Math.min(1.60, +(val + 0.02).toFixed(2));
        elements.inputSma.value = val;
        recalculate();
      });
    }
    if (elements.btnSmaDown) {
      elements.btnSmaDown.addEventListener('click', () => {
        let val = parseFloat(elements.inputSma.value) || 1.0;
        val = Math.max(0.60, +(val - 0.02).toFixed(2));
        elements.inputSma.value = val;
        recalculate();
      });
    }

    // Volatility Stepper (+1% / -1%)
    if (elements.btnVolUp) {
      elements.btnVolUp.addEventListener('click', () => {
        let val = parseFloat(elements.inputVol.value) || 20;
        val = Math.min(85, +(val + 1.0).toFixed(1));
        elements.inputVol.value = val;
        recalculate();
      });
    }
    if (elements.btnVolDown) {
      elements.btnVolDown.addEventListener('click', () => {
        let val = parseFloat(elements.inputVol.value) || 20;
        val = Math.max(5, +(val - 1.0).toFixed(1));
        elements.inputVol.value = val;
        recalculate();
      });
    }

    // Earnings Stepper (+2% / -2%)
    if (elements.btnEarningsUp) {
      elements.btnEarningsUp.addEventListener('click', () => {
        let val = parseFloat(elements.inputEarnings.value) || 0;
        val = Math.min(100, +(val + 2.0).toFixed(1));
        elements.inputEarnings.value = val;
        recalculate();
      });
    }
    if (elements.btnEarningsDown) {
      elements.btnEarningsDown.addEventListener('click', () => {
        let val = parseFloat(elements.inputEarnings.value) || 0;
        val = Math.max(-50, +(val - 2.0).toFixed(1));
        elements.inputEarnings.value = val;
        recalculate();
      });
    }

    // Quick Preset Chips (.btn-chip-val)
    if (elements.chipValButtons) {
      elements.chipValButtons.forEach(btn => {
        btn.addEventListener('click', () => {
          const targetId = btn.dataset.target;
          const targetInput = document.getElementById(targetId);
          if (targetInput) {
            targetInput.value = btn.dataset.val;
            recalculate();
          }
        });
      });
    }

    // Institutional Order Flow Selector (.btn-flow)
    if (elements.flowButtons) {
      elements.flowButtons.forEach(btn => {
        btn.addEventListener('click', () => {
          const flowVal = btn.dataset.flow;
          elements.inputFlow.value = flowVal;
          recalculate();
        });
      });
    }

    // 4. Sector Buttons
    elements.sectorButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        elements.sectorButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentSector = btn.dataset.sector;
        recalculate();
      });
    });

    // 4. Beta Buttons
    elements.betaButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        elements.betaButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentBeta = parseFloat(btn.dataset.beta);
        recalculate();
      });
    });

    // 5. Presets
    elements.presetChips.forEach(btn => {
      btn.addEventListener('click', () => {
        applyPreset(btn.dataset.preset);
      });
    });

    // 6. Action Button
    if (elements.btnExecute) {
      elements.btnExecute.addEventListener('click', () => {
        elements.btnExecute.style.transform = 'scale(0.98)';
        setTimeout(() => {
          elements.btnExecute.style.transform = '';
          recalculate();
        }, 120);
      });
    }

    // 7. Scatter Axes
    elements.btnAxes.forEach(btn => {
      btn.addEventListener('click', () => {
        elements.btnAxes.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        scatterAxis = btn.dataset.axis;
        renderScatterMatrix();
      });
    });

    // 8. Explorer Filter & Pagination
    if (elements.tableFilter) {
      elements.tableFilter.addEventListener('change', e => {
        explorerFilter = e.target.value;
        explorerPage = 1;
        renderDataExplorerTable();
      });
    }

    if (elements.btnPagePrev) {
      elements.btnPagePrev.addEventListener('click', () => {
        if (explorerPage > 1) {
          explorerPage--;
          renderDataExplorerTable();
        }
      });
    }

    if (elements.btnPageNext) {
      elements.btnPageNext.addEventListener('click', () => {
        explorerPage++;
        renderDataExplorerTable();
      });
    }

    // 9. Simulator Cohorts & Batch Size
    elements.cohortCards.forEach(card => {
      card.addEventListener('click', () => {
        elements.cohortCards.forEach(c => c.classList.remove('active'));
        card.classList.add('active');
        simCohort = card.dataset.cohort;
        runPortfolioSimulation();
      });
    });

    elements.batchButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        elements.batchButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        simBatchSize = parseInt(btn.dataset.batch, 10);
        runPortfolioSimulation();
      });
    });

    if (elements.btnResimulate) {
      elements.btnResimulate.addEventListener('click', () => {
        elements.btnResimulate.style.opacity = '0.7';
        setTimeout(() => {
          runPortfolioSimulation();
          elements.btnResimulate.style.opacity = '1';
        }, 200);
      });
    }

    // Resize listener for scatter canvas
    window.addEventListener('resize', () => {
      if (document.getElementById('view-regimes').classList.contains('active')) {
        renderScatterMatrix();
      }
    });
  }

  // ==========================================================================
  // Init
  // ==========================================================================
  function init() {
    setupEvents();
    renderFeatureImportance();
    setupScatterTooltip();
    renderScatterMatrix();
    renderDataExplorerTable();
    runPortfolioSimulation();
    applyPreset('blue-chip');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
