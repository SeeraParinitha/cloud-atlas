(() => {
 const el=id=>document.getElementById(id);
 let loaded=false, loading=null, historyPage=1, historyVersion=0, statsTimer;
 const titles={dashboard:'Dashboard',history:'Results history','how-it-works':'How it works',about:'About the project'};
 async function request(path,data,method){
  const response=await fetch(path,{...(data?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}:{}),...(method?{method}:{}),cache:'no-store'});
  const result=await response.json();
  if(!response.ok){if(response.status===401 && path!='/api/auth/login')signIn();throw Error(typeof result.detail==='string'?result.detail:'Unable to complete this request.');}
  return result;
 }
 function signIn(){clearInterval(statsTimer);el('boot').hidden=true;el('app-shell').hidden=true;el('signin-view').hidden=false;document.title='Sign in · Cloud Atlas';if(location.pathname!='/signin')history.replaceState({},'', '/signin');loadGoogleConfig();}
 async function loadWorkspace(){
  if(loaded)return;
  if(!loading)loading=new Promise((resolve,reject)=>{const script=document.createElement('script');script.src='/app.js';script.onload=()=>{loaded=true;resolve();};script.onerror=()=>{loading=null;reject(Error('Could not load the workspace. Refresh to retry.'));};document.body.append(script);});
  return loading;
 }
 function empty(container,title,description){container.replaceChildren();const box=document.createElement('div');box.className='empty-state';const symbol=document.createElement('span');symbol.textContent='◇';const heading=document.createElement('h3');heading.textContent=title;const text=document.createElement('p');text.textContent=description;box.append(symbol,heading,text);container.append(box);}
 function formatDate(value){const date=new Date(value);return Number.isNaN(+date)?'Date unavailable':date.toLocaleDateString(undefined,{year:'numeric',month:'short',day:'numeric',timeZone:'UTC'});}
 function cards(container,items){
  container.replaceChildren();
  if(!items.length){empty(container,'Your next discovery starts here.','Process an acquisition to save your first result.');return;}
  for(const item of items){
   const article=document.createElement('article');article.className='result-card';
   const image=document.createElement('img');image.src='/api/results/'+item.id+'/restored.png';image.alt='Saved satellite result';image.loading='lazy';
   const content=document.createElement('div');content.className='result-card-content';
   const title=document.createElement('h3');title.textContent=item.location_name||'Selected area';
   const coordinates=document.createElement('p');const b=item.bbox;coordinates.textContent=Array.isArray(b)&&b.length===4?`Area centre: ${((b[1]+b[3])/2).toFixed(4)}°, ${((b[0]+b[2])/2).toFixed(4)}°`:'Selected area';
   const saved=document.createElement('p');saved.textContent=formatDate(item.acquisition)+' · Sentinel-2 · Saved '+formatDate(item.saved_at);
   const button=document.createElement('button');button.textContent='Open comparison →';button.onclick=async()=>{button.disabled=true;try{await navigate('dashboard');await window.show(item);}catch(error){button.textContent='Could not open result. Retry';}finally{button.disabled=false;}};
   const remove=document.createElement('button');remove.textContent='Delete result';remove.className='delete-result';
   remove.onclick=async()=>{if(!confirm('Delete this saved result and its images? This cannot be undone.'))return;remove.disabled=true;try{await request('/api/results/'+item.id,undefined,'DELETE');document.getElementById('result').hidden=true;historyPage=1;await dashboard();if(location.pathname==='/history')await results();}catch(error){remove.disabled=false;alert(error.message);}};
   const rename=document.createElement('button');rename.textContent=item.location_name?'Edit location':'Add location';
   rename.onclick=async()=>{const name=prompt('Location name for this saved result:',item.location_name||'');if(name===null||!name.trim())return;rename.disabled=true;try{await request('/api/results/'+item.id+'/location',{location_name:name.trim()},'PATCH');await dashboard();if(location.pathname==='/history')await results();}catch(error){alert(error.message);}finally{rename.disabled=false;}};
   content.append(title,coordinates,saved,button,rename,remove);article.append(image,content);container.append(article);
  }
 }
 async function dashboard(){
  try{const data=await request('/api/history');el('stat-total').textContent=data.total;el('stat-active').textContent=data.active;cards(el('recent-results'),data.items.slice(0,3));}
  catch(error){el('stat-total').textContent='—';el('stat-active').textContent='—';empty(el('recent-results'),'History unavailable',error.message);}
 }
 async function results(){
  const version=++historyVersion;el('history-summary').textContent='Loading saved results…';
  const params=new URLSearchParams({q:el('history-query').value.trim(),date:el('history-date').value,page:String(historyPage)});
  try{const data=await request('/api/history?'+params);if(version!==historyVersion)return;cards(el('history-results'),data.items);el('history-summary').textContent=`${data.matched} saved result${data.matched===1?'':'s'} found`;el('history-page').textContent=`Page ${data.page} of ${Math.max(1,Math.ceil(data.matched/12))}`;el('history-prev').disabled=historyPage===1;el('history-next').disabled=historyPage*12>=data.matched;if(!data.items.length&&data.total)empty(el('history-results'),'No matching results','Try another location or clear the date filter.');}
  catch(error){if(version!==historyVersion)return;el('history-summary').textContent=error.message;empty(el('history-results'),'Could not load history','Use Search results to try again.');el('history-prev').disabled=true;el('history-next').disabled=true;}
 }
 async function navigate(page,replace=false){
  if(!titles[page])page='dashboard';
  el('boot').hidden=true;el('signin-view').hidden=true;el('app-shell').hidden=false;
  for(const view of document.querySelectorAll('[data-view]'))view.hidden=view.dataset.view!==page;
  for(const link of document.querySelectorAll('[data-page]')){const selected=link.dataset.page===page;link.classList.toggle('active',selected);if(selected)link.setAttribute('aria-current','page');else link.removeAttribute('aria-current');}
  el('page-crumb').textContent=titles[page];document.title=titles[page]+' · Cloud Atlas';
  if(location.pathname!='/'+page)history[replace?'replaceState':'pushState']({},'','/'+page);
  window.scrollTo(0,0);
  await loadWorkspace();
  if(page==='dashboard'){window.refreshAtlasMap?.();dashboard();}
  if(page==='history')results();
 }
 document.addEventListener('click',event=>{const link=event.target.closest('a[data-nav]');if(!link||event.ctrlKey||event.metaKey||event.shiftKey)return;event.preventDefault();navigate(link.getAttribute('href').slice(1)).catch(error=>{el('status').textContent=error.message;});});
 window.addEventListener('popstate',()=>{if(location.pathname==='/signin')signIn();else navigate(location.pathname.slice(1),true).catch(()=>signIn());});
 const authErrors={not_configured:'Google sign-in is not configured yet. Please contact the site owner.',cancelled:'Sign-in was cancelled. You can try again.',invalid_session:'Your sign-in request expired or could not be verified. Please try again.',failed:'Google sign-in could not be completed. Please try again.'};
 const initialError=new URLSearchParams(location.search).get('auth_error');
 if(initialError)el('login-error').textContent=authErrors[initialError]||'Please try signing in again.';
 async function loadGoogleConfig(){
  el('google-signin').disabled=true;
  try{const config=await request('/api/auth/config');el('google-signin').disabled=!config.google_enabled;if(!config.google_enabled)el('login-error').textContent=authErrors.not_configured;}
  catch{el('login-error').textContent='Unable to connect. Refresh this page to retry.';}
 }
 el('google-signin').onclick=()=>{el('google-signin').disabled=true;window.location.assign('/api/auth/google/start');};
 el('logout').onclick=async()=>{try{await request('/api/auth/logout',{});signIn();}catch(error){el('logout').textContent='Sign out failed. Retry';}};
 el('history-search').onclick=()=>{historyPage=1;results();};el('history-query').onkeydown=event=>{if(event.key==='Enter'){historyPage=1;results();}};el('history-prev').onclick=()=>{historyPage--;results();};el('history-next').onclick=()=>{historyPage++;results();};
 window.addEventListener('atlas-result-saved',dashboard);
 function startRefresh(){clearInterval(statsTimer);statsTimer=setInterval(()=>{if(!document.hidden && location.pathname==='/dashboard')dashboard();},30000);}
 request('/api/auth/me').then(async user=>{const avatar=el('account-avatar');avatar.textContent=(user.name||user.email||'?').trim().split(/\s+/).slice(0,2).map(part=>part[0]).join('').toUpperCase();avatar.title=user.name+' · '+user.email;await navigate(location.pathname.slice(1),true);startRefresh();}).catch(()=>signIn());
})();
