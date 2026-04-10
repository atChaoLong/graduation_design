// components.js - small UI component helpers (ES module)
import { formatPct, colorForConfidence } from './utils.js';

export function createChargeCard(charge){
  const wrap = document.createElement('div');
  wrap.className = 'charge-card';

  const left = document.createElement('div');
  left.style.display='flex';left.style.alignItems='center';left.style.gap='12px';
  const tag = document.createElement('div');
  tag.className = 'tag';
  tag.textContent = charge.name;
  tag.style.background = `linear-gradient(90deg, rgba(255,255,255,0.02), ${colorForConfidence(charge.confidence)})`;
  tag.style.color = '#001';
  tag.style.fontWeight = '700';
  left.appendChild(tag);

  const pr = document.createElement('div');
  pr.className = 'conf';
  const bar = document.createElement('i');
  pr.appendChild(bar);

  wrap.appendChild(left);
  wrap.appendChild(pr);

  // value label
  const val = document.createElement('div');
  val.textContent = formatPct(charge.confidence);
  val.style.marginLeft='12px';
  val.style.minWidth='52px';
  wrap.appendChild(val);

  // animate width
  requestAnimationFrame(()=>{ bar.style.width = `${Math.round(charge.confidence*100)}%`; });

  // float animation
  wrap.style.animation = 'floatIn .45s cubic-bezier(.2,.9,.2,1)';

  return wrap;
}

export function createArticleCard(article){
  const el = document.createElement('div');
  el.className = 'article';
  const left = document.createElement('div');
  left.innerHTML = `<strong>Article ${article.id}</strong> <div style="color:var(--muted);font-size:13px">${article.desc}</div>`;
  const right = document.createElement('div');
  right.style.color='var(--muted)';
  right.textContent = '🔗 View';
  el.appendChild(left);
  el.appendChild(right);
  return el;
}
