// utils.js - utility helpers (ES module)
export function sleep(ms){ return new Promise(r=>setTimeout(r, ms)); }

export function clamp(v, a=0, b=1){ return Math.max(a, Math.min(b, v)); }

export function formatPct(v){
  return `${Math.round(v*100)}%`;
}

export function colorForConfidence(conf){
  // green -> yellow -> red based on confidence 0..1
  const r = Math.round(255 * (1 - conf));
  const g = Math.round(200 * conf + 55*conf);
  const b = Math.round(220 * conf);
  return `rgb(${r},${g},${b})`;
}

const HISTORY_KEY = 'lexmind_history_v1';
export function saveHistory(text){
  if(!text) return;
  let list = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  list = [text, ...list.filter(t=>t!==text)].slice(0,12);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(list));
  return list;
}
export function loadHistory(){
  return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
}

// deterministic mock predictor: basic keyword mapping -> produce structured result
export function mockPredict(facts, threshold=0.5){
  // facts: array of strings
  const mapping = [
    {k:['偷','盗'], name:'盗窃罪', article:{id:'264', desc:'盗窃罪相关法条'}},
    {k:['抢','抢劫'], name:'抢劫罪', article:{id:'263', desc:'抢劫罪相关法条'}},
    {k:['诈骗','骗'], name:'诈骗罪', article:{id:'266', desc:'诈骗罪相关法条'}},
    {k:['故意伤害','伤害'], name:'故意伤害罪', article:{id:'234', desc:'故意伤害相关法条'}},
  ];
  const charges = {};
  const articles = {};
  facts.forEach(f=>{
    const lower = f.toLowerCase();
    mapping.forEach(m=>{
      m.k.forEach(kw=>{
        if(lower.includes(kw)){
          // bump a synthetic confidence based on keyword frequency and fact length
          const base = 0.6 + Math.min(0.35, (lower.split(kw).length - 1) * 0.15);
          const lenFactor = Math.min(0.25, Math.log1p(f.length)/20);
          const conf = clamp(base + lenFactor * (Math.random()*0.2+0.1), 0, 0.995);
          const prev = charges[m.name];
          charges[m.name] = Math.max(prev||0, conf);
          articles[m.article.id] = m.article;
        }
      });
    });
    // small chance to produce a fuzzy prediction
    if(Math.random() < 0.08){
      const m = mapping[Math.floor(Math.random()*mapping.length)];
      if(!charges[m.name]){
        charges[m.name] = 0.45 + Math.random()*0.35;
        articles[m.article.id] = m.article;
      }
    }
  });

  // fallback: if no keyword matched, return a soft top-1 prediction
  if(Object.keys(charges).length===0){
    const pick = mapping[Math.floor(Math.random()*mapping.length)];
    charges[pick.name] = 0.52 + Math.random()*0.3;
    articles[pick.article.id] = pick.article;
  }

  // filter by threshold and return arrays
  const chargesArr = Object.keys(charges).map(name=>({name, confidence: clamp(charges[name],0,1)}))
    .sort((a,b)=>b.confidence - a.confidence);
  const articlesArr = Object.keys(articles).map(k=>articles[k]);

  return {charges: chargesArr, articles: articlesArr};
}

// call backend API predict; returns JSON {charges: [...], articles: [...]}
export async function predictRemote(facts, threshold=0.5, timeout=8000){
  const controller = new AbortController();
  const id = setTimeout(()=>controller.abort(), timeout);
  try{
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({facts, threshold}),
      signal: controller.signal
    });
    clearTimeout(id);
    if(!res.ok){
      const txt = await res.text();
      throw new Error(`Server ${res.status}: ${txt}`);
    }
    return await res.json();
  }catch(e){
    clearTimeout(id);
    throw e;
  }
}
