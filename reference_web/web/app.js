const $=id=>document.getElementById(id);
let selectedPlaceName='';
let map,rectangle,marker,customBounds=null,scenes=[],searchBox,version=0,drawing=false,corner=null,busy=false,displayToken=0;
const today=()=>new Date().toISOString().slice(0,10);
for(const id of ['start','end'])$(id).max=today();
const now=new Date();$('end').value=now.toISOString().slice(0,10);now.setDate(now.getDate()-30);$('start').value=now.toISOString().slice(0,10);
function validate(b){const [w,s,e,n]=b;const width=(e-w)*111.32*Math.cos((s+n)/2*Math.PI/180),height=(n-s)*111.32;if(!b.every(Number.isFinite)||w< -180||e>180||s< -80||n>80||w>=e||s>=n||width>10||height>10)throw Error('Choose a rectangle no larger than 10 km on each side, away from the poles and date line.');return [width,height]}
function bounds(){if(customBounds)return customBounds.slice();const lat=+$('lat').value,lon=+$('lon').value,km=+$('km').value;if(!Number.isFinite(lat+lon+km)||km<.2||km>8||Math.abs(lat)>79||Math.abs(lon)>179)throw Error('Enter valid coordinates and an area width between 0.2 and 8 km.');const dy=km/2/111.32,dx=dy/Math.cos(lat*Math.PI/180);return [lon-dx,lat-dy,lon+dx,lat+dy]}
function invalidate(){displayToken++;$('result').hidden=true;version++;scenes=[];searchBox=null;$('restore').disabled=true;$('scene').disabled=true;$('scene').replaceChildren(new Option('Search this area for acquisitions',''));}
function renderArea(b){const [width,height]=validate(b);if(map){if(rectangle)rectangle.remove();rectangle=L.rectangle([[b[1],b[0]],[b[3],b[2]]],{color:'#54e3cd',weight:2,fillOpacity:.15}).addTo(map);const centre=[(b[1]+b[3])/2,(b[0]+b[2])/2];if(!marker){marker=L.marker(centre,{draggable:true,title:'Drag to move selected area'}).addTo(map);marker.on('dragend',()=>{try{const old=bounds(),c=marker.getLatLng(),dx=c.lng-(old[0]+old[2])/2,dy=c.lat-(old[1]+old[3])/2;selectArea([old[0]+dx,old[1]+dy,old[2]+dx,old[3]+dy]);}catch(e){renderArea(bounds());$('message').textContent=e.message;}});}else marker.setLatLng(centre);}$('area-info').textContent=`Selected area: ${width.toFixed(2)} km x ${height.toFixed(2)} km. Only this box will be downloaded.`;}
function selectArea(b){validate(b);selectedPlaceName='';customBounds=b.slice();$('lat').value=((b[1]+b[3])/2).toFixed(5);$('lon').value=((b[0]+b[2])/2).toFixed(5);invalidate();renderArea(b);$('message').textContent='Area changed. Find satellite images for this selection.';}
function draw(preserveName=false){if(preserveName!==true)selectedPlaceName='';customBounds=null;invalidate();try{renderArea(bounds())}catch(e){$('message').textContent=e.message}}
if(window.L){map=L.map('map').setView([18.5204,73.8567],12);L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'&copy; OpenStreetMap contributors',maxZoom:18}).addTo(map);map.on('click',e=>{if(busy)return;if(drawing){if(!corner){corner=e.latlng;$('area-info').textContent='Now click the opposite corner.';return;}const b=[Math.min(corner.lng,e.latlng.lng),Math.min(corner.lat,e.latlng.lat),Math.max(corner.lng,e.latlng.lng),Math.max(corner.lat,e.latlng.lat)];drawing=false;corner=null;$('draw-area').textContent='Draw a rectangle';try{selectArea(b)}catch(err){renderArea(bounds());$('message').textContent=err.message}return;}$('lat').value=e.latlng.lat.toFixed(5);$('lon').value=e.latlng.lng.toFixed(5);draw();});}else{$('map').textContent='Map unavailable. Enter coordinates to continue.';$('draw-area').disabled=true;}
for(const id of ['lat','lon'])$(id).addEventListener('change',()=>draw());
$('km').addEventListener('change',()=>draw(true));
for(const id of ['start','end'])$(id).addEventListener('change',()=>{invalidate();$('message').textContent='Dates changed. Find satellite images again.';});
$('draw-area').onclick=()=>{drawing=!drawing;corner=null;$('draw-area').textContent=drawing?'Cancel drawing':'Draw a rectangle';if(drawing){invalidate();$('area-info').textContent='Click one corner on the map, then the opposite corner (maximum 10 km per side).';}else renderArea(bounds());};draw();
async function api(path,data){const r=await fetch(path,data?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}:{});const body=await r.json();if(!r.ok)throw Error(typeof body.detail==='string'?body.detail:'Invalid request. Check the selected area and dates.');return body;}
api('/api/status').then(s=>{$('status').textContent=(s.model_ready?'Model ready':'Model missing')+' | '+(s.satellite_ready?'Satellite configured':'Satellite credentials needed')}).catch(e=>$('status').textContent=e.message);
let places=[];
$('find-place').onclick=async()=>{$('find-place').disabled=true;$('places').disabled=true;$('place-note').textContent='Searching places...';try{const q=$('place').value.trim();if(q.length<2)throw Error('Enter at least two characters.');const r=await api('/api/places',{query:q});places=r.places;$('places').replaceChildren(new Option('Choose a matching place',''));places.forEach((p,i)=>$('places').append(new Option(p.label,String(i))));$('places').disabled=!places.length;$('place-note').textContent=places.length?(r.suggestions?'Similar spellings found. Choose the intended place, then refine the box.':'Choose the correct place, then refine the box on the map.'):'No matching town found. Try a nearby city or select directly on the map.';}catch(e){$('place-note').textContent=e.message}finally{$('find-place').disabled=false;}};
$('place').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();if(!$('find-place').disabled)$('find-place').click();}};
$('places').onchange=()=>{if($('places').value==='')return;const p=places[+$('places').value];$('lat').value=p.lat;$('lon').value=p.lon;draw();selectedPlaceName=p.label;if(map)map.fitBounds(rectangle.getBounds(),{padding:[30,30],animate:false});};
async function coverageSearch(query,requestVersion){
 const response=await api('/api/search',query);
 if(!response.job_id)return response;
 for(let i=0;i<1200;i++){
  if(requestVersion!==version)return null;
  await new Promise(resolve=>setTimeout(resolve,1000));
  const job=await api('/api/jobs/'+response.job_id);
  if(requestVersion!==version)return null;
  if(job.status==='failed')throw Error(job.error);
  if(job.status==='complete')return job.result;
  $('message').textContent=job.message||'Checking image coverage...';
 }
 throw Error('Coverage checks are still running. Try a shorter date range if needed.');
}
$('search').onclick=async()=>{$('search').disabled=true;invalidate();const requestVersion=version;$('message').textContent='Searching acquisitions...';try{if(drawing)throw Error('Finish or cancel drawing first.');if($('start').value>today()||$('end').value>today())throw Error('Dates cannot be in the future (UTC).');if($('start').value>$('end').value)throw Error('From date must be before To date.');const box=bounds();validate(box);const r=await coverageSearch({bbox:box,start:$('start').value,end:$('end').value},requestVersion);if(!r||requestVersion!==version)return;searchBox=box;scenes=r.scenes;$('scene').replaceChildren();scenes.forEach((s,i)=>$('scene').append(new Option(new Date(s.datetime).toLocaleString()+' | '+(s.cloud_cover==null?'unknown cloud cover':Number(s.cloud_cover).toFixed(1)+'% scene cloud'),String(i))));$('scene').disabled=!scenes.length;$('restore').disabled=!scenes.length;$('message').textContent=scenes.length?scenes.length+' images with full area coverage found.'+(r.truncated?' Narrow the dates to check more images.':''):'No images with sufficient area coverage found. Try another date range or a smaller area.';if(r.coverage_rejected?.length)$('message').textContent+=` ${r.coverage_rejected.length} partial images excluded.`;if(r.coverage_failed?.length)$('message').textContent+=` ${r.coverage_failed.length} images could not be checked.`;}catch(e){$('message').textContent=e.message}finally{$('search').disabled=false;}};
function setBusy(value){busy=value;for(const id of ['find-place','place','places','draw-area','lat','lon','km','start','end','search','scene','restore'])$(id).disabled=value;if(marker)value?marker.dragging.disable():marker.dragging.enable();if(!value){$('places').disabled=!places.length;$('scene').disabled=!scenes.length;$('restore').disabled=!scenes.length;}}
$('restore').onclick=async()=>{if(!searchBox||!scenes.length)return;const chosen=scenes[+$('scene').value],box=searchBox.slice();setBusy(true);$('result').hidden=true;$('message').textContent='Queued for restoration...';try{const job=await api('/api/restore',{bbox:box,acquisition:chosen.datetime,location_name:selectedPlaceName});for(let i=0;i<600;i++){await new Promise(r=>setTimeout(r,2000));const j=await api('/api/jobs/'+job.job_id);$('message').textContent=j.status==='running'?'Processing image...':'Waiting for the model...';if(j.status==='failed')throw Error(j.error);if(j.status==='complete'){await show(j.result);$('message').textContent='Result ready.';window.dispatchEvent(new Event('atlas-result-saved'));return;}}throw Error('Job still pending. Avoid submitting it again; check server status.');}catch(e){$('message').textContent=e.message}finally{setBusy(false);}};
async function loadPhoto(element,url){
 element.src=url;
 try{await element.decode();}catch(e){throw Error('The result image could not be loaded. Retry when the connection is available.');}
}
async function show(r){
 const token=++displayToken,base='/api/results/'+r.id+'/',observed=r.display_mode==='full_reference';
 $('result').hidden=true;
 $('comparison').hidden=false;
 $('after').style.clipPath='inset(0 0 0 0)';$('slider').value=100;
 await Promise.all([loadPhoto($('before'),base+'original.png'),loadPhoto($('after'),base+'restored.png')]);
 if(token!==displayToken)return;
 $('result-label').textContent='Result';
 $('after').alt=observed?'Earlier observation of the selected area':'Processed satellite image';
 const incomplete=r.result_available===false||['reference_preview_shown','full_reference_unavailable','partial_reconstruction','no_usable_reference'].includes(r.state);
 $('outcome').hidden=!incomplete;
 $('outcome').textContent=incomplete?(observed?'Best available reference — some clouds or shadows may remain.':r.display_mode==='original'?'Original image shown; a usable reconstruction could not be produced.':'Best available model preview — some clouds or shadows may remain.'):'';
 $('downloads').replaceChildren();
 for(const [name,label] of [['original.png','Original PNG'],['restored.png','Result PNG']]){
  const a=document.createElement('a');a.href=base+name;a.download=name;a.textContent=label;$('downloads').append(a);
 }
 $('result').hidden=false;$('result').scrollIntoView({behavior:'auto'});
}
$('slider').oninput=()=>{$('after').style.clipPath='inset(0 '+(100-$('slider').value)+'% 0 0)';};

window.refreshAtlasMap=()=>{if(map)setTimeout(()=>map.invalidateSize(),50);};
