// app.js - core logic & wiring (ES module)
import { mockPredict, predictRemote, saveHistory, loadHistory, sleep } from './utils.js';
import { createChargeCard, createArticleCard } from './components.js';

const factsEl = document.getElementById('facts');
const runBtn = document.getElementById('runBtn');
const clearBtn = document.getElementById('clearBtn');
const loadingArea = document.getElementById('loadingArea');
const outputArea = document.getElementById('outputArea');
const resultPanel = document.getElementById('resultPanel');
const emptyNotice = document.getElementById('emptyNotice');
const chargesEl = document.getElementById('charges');
const articlesEl = document.getElementById('articles');
const thresholdEl = document.getElementById('threshold');
const thresholdVal = document.getElementById('threshold-val');
const historyEl = document.getElementById('history');

thresholdEl.addEventListener('input', ()=>thresholdVal.textContent = parseFloat(thresholdEl.value).toFixed(2));

function showLoading(){
  loadingArea.classList.remove('hidden');
  outputArea.classList.add('hidden');
  emptyNotice.classList.add('hidden');
}
function showOutput(){
  loadingArea.classList.add('hidden');
  outputArea.classList.remove('hidden');
  emptyNotice.classList.add('hidden');
}
function showEmpty(){
  loadingArea.classList.add('hidden');
  outputArea.classList.add('hidden');
  emptyNotice.classList.remove('hidden');
}

// render history
function renderHistory(){
  const list = loadHistory();
  historyEl.innerHTML = '<div style="font-weight:700;margin-bottom:6px">Input History</div>';
  list.forEach(t=>{
    const item = document.createElement('div');
    item.className = 'hist-item';
    item.textContent = t.length>120? t.slice(0,120)+'…': t;
    item.onclick = ()=>{ factsEl.value = t; };
    historyEl.appendChild(item);
  });
}
renderHistory();

async function runPrediction(){
  const raw = factsEl.value.trim();
  if(!raw) return;
  saveHistory(raw);
  renderHistory();

  const lines = raw.split('\n').map(s=>s.trim()).filter(Boolean);
  const threshold = parseFloat(thresholdEl.value);
  showLoading();

  // simulate loading progression & AI thinking animation
  const loaderText = loadingArea.querySelector('.loader-text');
  let elapsed = 0;
  const simulate = async ()=>{
    while(!window._predictionDone){
      loaderText.textContent = ['AI thinking', 'AI thinking .', 'AI thinking ..', 'AI thinking ...'][Math.floor((elapsed/600)%4)];
      await sleep(400);
      elapsed += 400;
    }
  };
  window._predictionDone = false;
  simulate();

  // progress skeleton duration randomized to feel like real compute
  const duration = 1200 + Math.random()*1200;
  const start = performance.now();
  while(performance.now() - start < duration){
    await sleep(120);
  }

  // call backend; fall back to mock if backend fails
  let result;
  try{
    result = await predictRemote(lines, threshold);
    if(result && result.error){
      throw new Error(result.error || 'backend error');
    }
  }catch(e){
    console.warn('Remote predict failed, falling back to local mock:', e);
    result = mockPredict(lines, threshold);
  }

  // small stagger to emulate streaming / typing
  window._predictionDone = true;
  await sleep(210);

  // render results with animations
  chargesEl.innerHTML = '';
  articlesEl.innerHTML = '';

  // typing card reveal style: reveal names and then confidences animate
  for(let c of result.charges){
    const card = createChargeCard(c);
    chargesEl.appendChild(card);
    await sleep(120);
  }
  for(let a of result.articles){
    const art = createArticleCard(a);
    articlesEl.appendChild(art);
    await sleep(80);
  }

  showOutput();
  // animate advantage bars on first result
  document.querySelectorAll('.adv .bar').forEach(b=>{
    const v = parseInt(b.parentElement.dataset ? b.parentElement.dataset.value : b.parentElement.getAttribute('data-value')) || b.parentElement.getAttribute('data-value') || 80;
    const fill = b.querySelector('.bar-fill');
    const target = b.parentElement.dataset ? b.parentElement.dataset.value : b.parentElement.getAttribute('data-value');
    fill.style.width = `${target || 80}%`;
  });
}

// attach events
runBtn.addEventListener('click', async ()=>{
  runBtn.disabled = true;
  runBtn.textContent = 'Running...';
  try{
    await runPrediction();
  }catch(e){
    console.error(e);
    alert('Prediction failed: ' + e.message);
  }finally{
    runBtn.disabled = false;
    runBtn.textContent = 'Run Prediction';
  }
});
clearBtn.addEventListener('click', ()=>{ factsEl.value = ''; showEmpty(); });

// keyboard shortcut
factsEl.addEventListener('keydown', (e)=>{
  if((e.ctrlKey||e.metaKey) && e.key === 'Enter'){ runBtn.click(); }
});

// small particle canvas for hero
(function heroCanvas(){
  const canvas = document.getElementById('hero-canvas');
  const ctx = canvas.getContext('2d');
  let w = canvas.width = innerWidth;
  let h = canvas.height = 420;
  const pts = [];
  for(let i=0;i<50;i++){
    pts.push({x:Math.random()*w,y:Math.random()*h,vx:(Math.random()-0.5)*0.2,vy:(Math.random()-0.5)*0.2,r:Math.random()*2+1});
  }
  function loop(){
    ctx.clearRect(0,0,w,h);
    // gradient background subtle
    const g = ctx.createLinearGradient(0,0,w,h);
    g.addColorStop(0,'rgba(108,99,255,0.03)');
    g.addColorStop(1,'rgba(0,212,255,0.02)');
    ctx.fillStyle = g;
    ctx.fillRect(0,0,w,h);

    for(let p of pts){
      p.x += p.vx; p.y += p.vy;
      if(p.x<0||p.x>w) p.vx *= -1;
      if(p.y<0||p.y>h) p.vy *= -1;
    }
    // draw connections
    for(let i=0;i<pts.length;i++){
      for(let j=i+1;j<pts.length;j++){
        const a=pts[i], b=pts[j];
        const d = Math.hypot(a.x-b.x, a.y-b.y);
        if(d < 100){
          ctx.beginPath();
          ctx.strokeStyle = `rgba(100,120,255,${1 - d/120})`;
          ctx.lineWidth = 0.6;
          ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
        }
      }
    }
    for(let p of pts){
      ctx.beginPath();
      ctx.fillStyle = 'rgba(180,200,255,0.9)';
      ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fill();
    }
    requestAnimationFrame(loop);
  }
  loop();
  addEventListener('resize', ()=>{ w = canvas.width = innerWidth; });
})();

// init small UI polish
document.addEventListener('DOMContentLoaded', ()=>{
  showEmpty();
});
