'use strict';
// Associate every control with a visible label, including the original coordinate inputs.
document.querySelectorAll('.control-group').forEach(group=>{const label=group.querySelector('label'),el=group.querySelector('input,select');if(label&&el)label.htmlFor=el.id});
const labelText={'lat1':'Start latitude °N','lon1':'Start longitude °E','lat2':'End latitude °N','lon2':'End longitude °E','val-start':'Observations from','val-end':'Observations through'};
for(const [id,text] of Object.entries(labelText)){const el=$(id),label=guidanceNode('label','coordinate-label',text);label.htmlFor=id;el.before(label);label.append(el);if(id.startsWith('lat')){el.min=5;el.max=30}else if(id.startsWith('lon')){el.min=45;el.max=105}}
$('date-b').labels[0].textContent='Date B · compare with';
$('map-view').onchange=()=>safe(render);
for(const id of ['location-preset','time-place'])$(id).onchange=()=>{if($(id).value==='custom')return;const [lat,lon]=$(id).value.split(',');$('lat').value=lat;$('lon').value=lon;$('time-lat').value=lat;$('time-lon').value=lon;syncLocationPresets();$('status').textContent='Location selected — select Show results to update.'};
for(const id of ['lat','lon'])$(id).addEventListener('input',()=>{syncLocationPresets();$('status').textContent='Selections changed — select Show this location.'});
for(const id of ['time-lat','time-lon'])$(id).addEventListener('input',()=>{$('time-place').value='custom';$('status').textContent='Location changed — select Show time charts.'});
$('route-preset').onchange=()=>{if($('route-preset').value==='custom')return;$('route-preset').value.split(',').forEach((v,i)=>$( ['lat1','lon1','lat2','lon2'][i]).value=v);$('status').textContent='Route selected — select Show cross-section.'};
for(const id of ['lat1','lon1','lat2','lon2'])$(id).addEventListener('input',()=>{$('route-preset').value='custom'});
for(const id of ['date','date-b','depth','cmin','cmax','start','end','val-start','val-end','diff-range','lat1','lon1','lat2','lon2'])$(id).addEventListener('change',()=>{$('status').textContent='Selections changed — apply them to refresh the charts.'});
$('diff-range').addEventListener('change',()=>{if(currentTab==='compare')safe(render)});
document.querySelector('a[href="#tab-validation"]')?.addEventListener('click',e=>{e.preventDefault();safe(()=>activate('validation'));$('workspace').scrollIntoView({behavior:'smooth'})});
// Clarify the selected dataset without implying that historical data is live.
const modeDescription=guidanceNode('p','mode-explanation','Dataset status appears above. Retrospective means historical data; synthetic means a generated demonstration.');document.querySelector('.app-header').after(modeDescription);
document.querySelectorAll('button[data-tab]').forEach(button=>button.addEventListener('click',()=>{$('workspace').scrollIntoView({behavior:'smooth',block:'start'})}));
// Add definitions to sample counts where first-time readers encounter them.
for(const [id,text] of Object.entries({'n-pairs':'Individual prediction–measurement comparisons across depths.','n-profiles':'Measurement profiles, each containing several depths at one place and time.','n-floats':'Distinct Argo instruments represented in these comparisons.'})){const note=guidanceNode('p','sample-definition',text);$(id).closest('.stat-box').append(note)}
