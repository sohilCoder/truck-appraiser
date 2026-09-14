/* TruckVal v2 — frontend */
(function () {
  "use strict";

  /* ---- i18n (same as v1, trimmed) ---- */
  var LANG = {
    en: {
      appName:"TruckVal", appSub:"Commercial Truck Valuation · v2",
      statusReady:"Ready", statusBusy:"Analysing photo…", startOver:"Start over",
      introBadge:"GP-Powered Pricing", introTitle:"What is this truck worth — and what's wrong with it?",
      introDesc:"Upload photos of any commercial truck. Features are extracted by AI, then priced by a Gaussian Process trained on real market comparables.",
      classPickup:"Pickup", classSemi:"Semi / 18-Wheeler", classBox:"Box Truck",
      classFlatbed:"Flatbed", classDump:"Dump Truck", classMore:"+ More",
      dropTitle:"Add truck photos", dropDesc:"Drag & drop or choose files · JPEG, PNG, WebP, HEIC",
      dropHint:"Bad lighting, mud and awkward angles are fine.",
      choosePhotos:"Choose photos",
      disclaimer:"Estimates from photographs and curated market data. Not an appraisal or guarantee of value.",
      coverageTitle:"Inspection Coverage", coverageEmpty:"Coverage appears after the first photo.",
      detailsTitle:"Known Details", optional:"Optional",
      fieldYear:"Year", fieldMake:"Make", fieldModel:"Model",
      fieldTrim:"Trim / Spec", fieldOdometer:"Odometer", fieldUnits:"Units",
      miles:"Miles", km:"Kilometres", applyDetails:"Apply details",
      specsNote:"Typed details override vision guesses and sharpen the GP estimate.",
      verdictLabel:"GP price estimate", verdictMidpoint:"Midpoint", verdictBand:"band",
      confidenceLabel:"Confidence", coverageScoreLabel:"Coverage",
      nextPhotoLabel:"Most useful next photo",
      compsTitle:"Nearest market comparables", compsEmpty:"Comparables appear once a truck is identified.",
      featTitle:"Extracted features", featEmpty:"Features extracted after first accepted photo.",
      noEstimate:"No estimate yet",
      noEstimateReason_noPhotos:"Add at least one usable photo of a commercial truck.",
      noEstimateReason_rejected:"No photos show a commercial truck clearly enough to price.",
      noEstimateReason_error:"Pricing error — check terminal output.",
      noEstimateReason_waiting:"Waiting on analysis…",
      readingPhoto:"Reading photo", removePhoto:"Remove",
      photoNotUsed:"Not used", photoAccepted_none:"No defects seen",
      photoAccepted_one:"1 finding", photoAccepted_many:"findings",
      covSeen:"Seen", covPartial:"Partial", covMissing:"Not seen",
      methodNote:"Price estimated by Gaussian Process regression over curated comparable listings. The vision model extracts features; the GP predicts price from the market neighbourhood of those features.",
    },
    tr: {
      appName:"TruckVal", appSub:"Ticari Araç Değerleme · v2",
      statusReady:"Hazır", statusBusy:"Fotoğraf analiz ediliyor…", startOver:"Yeniden başla",
      introBadge:"GP Destekli Fiyatlandırma", introTitle:"Bu kamyon ne kadar eder — ve nesi var?",
      introDesc:"Herhangi bir ticari aracın fotoğraflarını yükleyin. Özellikler YZ tarafından çıkarılır, ardından gerçek piyasa verileriyle eğitilmiş bir Gaussian Process ile fiyatlandırılır.",
      classPickup:"Pikap", classSemi:"TIR / 18 Teker", classBox:"Kamyon",
      classFlatbed:"Açık Kasa", classDump:"Damperli", classMore:"+ Daha fazla",
      dropTitle:"Araç fotoğrafları ekle", dropDesc:"Sürükle & bırak veya dosya seç · JPEG, PNG, WebP, HEIC",
      dropHint:"Kötü aydınlatma, çamur ve tuhaf açılar sorun değil.",
      choosePhotos:"Fotoğraf seç",
      disclaimer:"Tahminler fotoğraflar ve piyasa verilerinden oluşturulur. Ekspertiz veya değer garantisi değildir.",
      coverageTitle:"Muayene Kapsamı", coverageEmpty:"Kapsam, ilk fotoğraf analiz edildikten sonra görünür.",
      detailsTitle:"Bilinen Detaylar", optional:"İsteğe bağlı",
      fieldYear:"Yıl", fieldMake:"Marka", fieldModel:"Model",
      fieldTrim:"Donanım", fieldOdometer:"Km sayacı", fieldUnits:"Birim",
      miles:"Mil", km:"Kilometre", applyDetails:"Detayları uygula",
      specsNote:"Yazılan detaylar görsel tahminlerin önüne geçer.",
      verdictLabel:"GP fiyat tahmini", verdictMidpoint:"Orta nokta", verdictBand:"bant",
      confidenceLabel:"Güven", coverageScoreLabel:"Kapsam",
      nextPhotoLabel:"En yararlı sonraki fotoğraf",
      compsTitle:"En yakın piyasa karşılaştırmaları", compsEmpty:"Karşılaştırmalar araç tanımlandıktan sonra görünür.",
      featTitle:"Çıkarılan özellikler", featEmpty:"Özellikler ilk kabul edilen fotoğraftan sonra görünür.",
      noEstimate:"Henüz tahmin yok",
      noEstimateReason_noPhotos:"En az bir ticari araç fotoğrafı ekleyin.",
      noEstimateReason_rejected:"Hiçbir fotoğraf yeterince net bir ticari araç göstermiyor.",
      noEstimateReason_error:"Fiyatlandırma hatası — terminal çıktısını kontrol edin.",
      noEstimateReason_waiting:"Analiz bekleniyor…",
      readingPhoto:"Fotoğraf okunuyor", removePhoto:"Kaldır",
      photoNotUsed:"Kullanılmadı", photoAccepted_none:"Kusur görülmedi",
      photoAccepted_one:"1 bulgu", photoAccepted_many:"bulgu",
      covSeen:"Görüldü", covPartial:"Kısmi", covMissing:"Görülmedi",
      methodNote:"Fiyat, seçilmiş karşılaştırmalı listeler üzerinde Gaussian Process regresyonu ile tahmin edilir.",
    },
  };

  var currentLang = "en";
  function t(k) { return (LANG[currentLang] || LANG.en)[k] || LANG.en[k] || k; }
  function applyI18n() {
    document.querySelectorAll("[data-i18n]").forEach(function(el) {
      var v = t(el.getAttribute("data-i18n")); if (v) el.textContent = v;
    });
    document.documentElement.lang = currentLang;
  }

  /* ---- state ---- */
  var MAX_EDGE = 1280, JPEG_Q = 0.82;
  var state = { sessionId:null, session:null, thumbs:{}, pending:[], busy:false, queue:[], theme:"dark" };
  var el = {};

  /* ---- utils ---- */
  function $(id) { return document.getElementById(id); }
  function esc(v) {
    return String(v===null||v===undefined?"":v)
      .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
  }
  function money(v) {
    if (v===null||v===undefined||isNaN(v)) return "—";
    return (v<0?"−":"") + "$" + Math.abs(Math.round(v)).toLocaleString("en-US");
  }
  function titleize(v) { return String(v||"").replace(/_/g," ").replace(/^\w/,function(c){return c.toUpperCase();}); }

  var VIEW_LABELS = {
    front:"Front",rear:"Rear",driver_side:"Driver side",passenger_side:"Passenger side",
    front_three_quarter:"Front ¾",rear_three_quarter:"Rear ¾",tire:"Tire",
    interior:"Cab interior",dashboard:"Dashboard",odometer:"Odometer",
    sleeper:"Sleeper",fifth_wheel:"Fifth wheel",cargo_area:"Cargo area",
    bed:"Truck bed",engine_bay:"Engine bay",undercarriage:"Underside",
    detail:"Detail",unknown:"Unidentified view",
  };

  /* ---- api ---- */
  function api(method, path, body) {
    return fetch(path,{method:method,headers:body?{"Content-Type":"application/json"}:{},
      body:body?JSON.stringify(body):undefined})
    .then(function(r){
      return r.json().catch(function(){throw new Error("Server response unreadable.");})
      .then(function(d){if(!r.ok)throw new Error(d.error||"Request failed.");return d;});
    });
  }

  /* ---- images ---- */
  function readFile(f){
    return new Promise(function(res,rej){
      var r=new FileReader();
      r.onload=function(){res(r.result);}; r.onerror=function(){rej(new Error("Could not read "+f.name));};
      r.readAsDataURL(f);
    });
  }
  function resize(dataUrl){
    return new Promise(function(res){
      var img=new Image();
      img.onload=function(){
        var sc=Math.min(1,MAX_EDGE/Math.max(img.width,img.height));
        if(sc===1&&dataUrl.length<3200000)return res(dataUrl);
        var c=document.createElement("canvas");
        c.width=Math.round(img.width*sc);c.height=Math.round(img.height*sc);
        c.getContext("2d").drawImage(img,0,0,c.width,c.height);
        try{res(c.toDataURL("image/jpeg",JPEG_Q));}catch(e){res(dataUrl);}
      };
      img.onerror=function(){res(dataUrl);}; img.src=dataUrl;
    });
  }

  /* ---- upload ---- */
  function enqueue(files){
    var acc=[];
    for(var i=0;i<files.length;i++){
      var f=files[i];
      if(f.type&&f.type.indexOf("image/")!==0){flash("\u201c"+f.name+"\u201d is not an image file.");continue;}
      acc.push(f);
    }
    if(!acc.length)return;
    state.queue=state.queue.concat(acc); drain();
  }
  function drain(){
    if(state.busy||!state.queue.length||!state.sessionId)return;
    var file=state.queue.shift(); state.busy=true;
    setStatus("busy",t("statusBusy"));
    var entry={key:"p_"+Math.random().toString(36).slice(2,9),thumb:null,filename:file.name};
    state.pending.push(entry); render();
    readFile(file).then(resize).then(function(dataUrl){
      entry.thumb=dataUrl; render();
      return api("POST","/api/sessions/"+state.sessionId+"/photos",{data_url:dataUrl,filename:file.name})
      .then(function(s){if(s.uploaded_photo)state.thumbs[s.uploaded_photo.id]=dataUrl;state.session=s;});
    })
    .catch(function(err){flash(err.message||"Upload failed.");})
    .then(function(){
      state.pending=state.pending.filter(function(p){return p.key!==entry.key;});
      state.busy=false; render();
      if(state.queue.length)drain();
      else setStatus("ready",state.session&&state.session.mock_mode?"Offline demo":t("statusReady"));
    });
  }

  /* ---- status ---- */
  function setStatus(kind,text){
    el.status.className="status"+(kind==="busy"?" status--busy":kind==="down"?" status--down":"");
    el.statusText.textContent=text;
  }
  var flashes=[];
  function flash(msg){
    flashes.push(msg); renderAlerts();
    setTimeout(function(){flashes=flashes.filter(function(m){return m!==msg;});renderAlerts();},7000);
  }
  function renderAlerts(){
    var p=[];
    flashes.forEach(function(m){p.push('<div class="alert alert--error">'+esc(m)+"</div>");});
    if(state.session&&state.session.error)p.push('<div class="alert alert--warn">'+esc(state.session.error)+"</div>");
    if(state.session&&state.session.mock_mode)p.push('<div class="alert alert--warn">Offline mock mode — results are canned.</div>');
    el.alerts.innerHTML=p.length?'<div style="margin-top:14px">'+p.join("")+"</div>":"";
  }

  /* ---- render hub ---- */
  function render(){
    renderPhotos(); renderVerdict(); renderAsk(); renderCoverage();
    renderComps(); renderFeatures(); renderAlerts();
  }

  /* ---- photos ---- */
  function renderPhotos(){
    var session=state.session,cards=[];
    if(session)session.photos.forEach(function(p){cards.push(photoCard(p));});
    state.pending.forEach(function(e){cards.push(pendingCard(e));});
    el.photoList.innerHTML=cards.join("");
    el.photoList.querySelectorAll("[data-remove]").forEach(function(node){
      node.addEventListener("click",function(){removePhoto(node.getAttribute("data-remove"));});
    });
  }
  function pendingCard(e){
    return '<article class="photo"><div class="photo__shot">'+(e.thumb?'<img src="'+esc(e.thumb)+'" alt="">':"")+
      '<div class="photo__scan"><div class="scanner"></div><span>'+esc(t("readingPhoto"))+"</span></div></div>"+
      '<div class="photo__body"><div class="photo__top"><span class="photo__view">'+esc(t("readingPhoto"))+"</span></div></div></article>";
  }
  function photoCard(photo){
    var thumb=state.thumbs[photo.id],flagged=!photo.accepted,findings=photo.findings||[],count=findings.length;
    var countLabel=count===0?t("photoAccepted_none"):count===1?t("photoAccepted_one"):count+" "+t("photoAccepted_many");
    var chips=findings.map(function(f){
      return '<span class="chip"><span class="sev sev--'+f.severity+'"></span>'+esc(titleize(f.type))+" — "+esc(f.area)+"</span>";
    }).join("");
    var body=[];
    body.push('<div class="photo__top">');
    body.push('<span class="photo__view">'+esc(VIEW_LABELS[photo.view]||"Photo")+"</span>");
    body.push(flagged?'<span class="tag tag--fail">'+esc(t("photoNotUsed"))+"</span>":'<span class="tag tag--pass">'+esc(countLabel)+"</span>");
    if(photo.mileage)body.push('<span class="tag">'+Number(photo.mileage).toLocaleString("en-US")+" "+(photo.mileage_units||"mi")+"</span>");
    if(photo.make_guess)body.push('<span class="tag">'+esc(photo.make_guess+(photo.model_guess?" "+photo.model_guess:""))+(photo.year_guess?" "+photo.year_guess:"")+"</span>");
    body.push('<button class="photo__remove" type="button" data-remove="'+esc(photo.id)+'">'+esc(t("removePhoto"))+"</button>");
    body.push("</div>");
    if(flagged){
      body.push('<p class="photo__reject">'+esc(photo.rejection_reason||"This photo could not be used.")+"</p>");
      if(photo.guidance)body.push('<p class="photo__guidance">'+esc(photo.guidance)+"</p>");
    } else {
      if(chips)body.push('<div class="chips">'+chips+"</div>");
      if(photo.assessment_notes)body.push('<p class="photo__notes">'+esc(photo.assessment_notes)+"</p>");
    }
    return '<article class="photo'+(flagged?" is-flagged":"")+'" id="card-'+esc(photo.id)+'">'+
      '<div class="photo__shot">'+(thumb?'<img src="'+esc(thumb)+'" alt="">':"")+
      '</div><div class="photo__body">'+body.join("")+"</div></article>";
  }
  function removePhoto(pid){
    api("DELETE","/api/sessions/"+state.sessionId+"/photos/"+pid)
    .then(function(s){state.session=s;delete state.thumbs[pid];render();})
    .catch(function(err){flash(err.message);});
  }

  /* ---- verdict ---- */
  function renderVerdict(){
    var session=state.session,v=session&&session.valuation;
    if(!v){
      el.verdict.innerHTML='<div class="panel"><div class="panel__body"><div class="empty">'+
        '<div class="empty__ring"><svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16.1v.1"/></svg></div>'+
        "<h3>"+esc(t("noEstimate"))+"</h3><p>"+esc(emptyReason())+"</p></div></div></div>";
      return;
    }
    var lo=Math.min(v.price_low,v.price_low*.88),hi=Math.max(v.price_high,v.price_high*1.06);
    var span=Math.max(1,hi-lo),pct=function(x){return((x-lo)/span*100).toFixed(2);};
    var bl=pct(v.price_low),bw=Math.max(2,pct(v.price_high)-bl);
    var confLow=v.confidence<45,confClass=confLow?"mtr__progress--low":"mtr__progress--hot";
    var confVC=confLow?"mtr__value--low":"mtr__value--hot";
    el.verdict.innerHTML='<div class="verdict">'+
      '<div class="verdict__hero">'+
        '<div class="verdict__label">'+esc(t("verdictLabel"))+"</div>"+
        '<div class="verdict__range num">'+money(v.price_low)+" – "+money(v.price_high)+"</div>"+
        '<div class="verdict__point num">'+t("verdictMidpoint")+" "+money(v.price_point)+" &middot; "+v.confidence+"% confidence</div>"+
      "</div>"+
      '<div class="vscale"><div class="vscale__labels"><span class="num">'+money(lo)+'</span><span class="num">'+money(hi)+"</span></div>"+
        '<div class="vscale__track"><div class="vscale__rail"></div>'+
        '<div class="vscale__fill" style="left:'+bl+"%;width:"+bw+'%"></div></div></div>'+
      '<div class="verdict__meters">'+
        mtr(t("confidenceLabel"),v.confidence+"%",v.confidence,confClass,confVC,
          confLow?"More photos will raise confidence.":"Based on "+v.n_comps_used+" comparable listings.")+
        mtr(t("coverageScoreLabel"),v.coverage_score+"%",v.coverage_score,"mtr__progress","mtr__value--cool",covNote())+
      "</div></div>";
  }
  function mtr(name,val,pct,barCls,valCls,note){
    return "<div><div class='mtr__header'><span class='mtr__name'>"+esc(name)+"</span>"+
      "<span class='mtr__value "+valCls+" num'>"+esc(val)+"</span></div>"+
      "<div class='mtr__bar'><div class='"+barCls+"' style='width:"+Math.max(2,Math.min(100,pct))+"%'></div></div>"+
      (note?"<div class='mtr__note'>"+esc(note)+"</div>":"")+"</div>";
  }
  function covNote(){
    var s=state.session;if(!s||!s.coverage)return"";
    var m=s.coverage.categories.filter(function(c){return c.status==="missing";});
    if(!m.length)return"Every inspection area has evidence.";
    var names=m.slice(0,3).map(function(c){return c.label.toLowerCase();});
    return"Still unseen: "+names.join(", ")+(m.length>3?" +"+( m.length-3):".")+".";
  }
  function emptyReason(){
    var s=state.session;
    if(!s||!s.photos.length)return t("noEstimateReason_noPhotos");
    if(s.accepted_count===0)return t("noEstimateReason_rejected");
    if(s.error)return t("noEstimateReason_error");
    return t("noEstimateReason_waiting");
  }

  /* ---- ask ---- */
  function renderAsk(){
    var s=state.session;
    if(!s||!s.recommendation||s.accepted_count===0){el.ask.innerHTML="";return;}
    var rec=s.recommendation;
    var queue=(s.next_priorities||[]).slice(1,4).map(function(p){
      return'<span class="tag">'+esc(p.label||p.view)+"</span>";
    }).join("");
    el.ask.innerHTML='<div class="ask">'+
      '<div class="ask__label"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 4h-5L8 6H4a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-4l-1.5-2Z"/><circle cx="12" cy="13" r="3.5"/></svg>'+
        "<span>"+esc(t("nextPhotoLabel"))+"</span></div>"+
      '<p class="ask__text">'+esc(rec.ask||rec.instruction||"")+"</p>"+
      '<p class="ask__why">'+esc(rec.reason||"")+"</p>"+
      (rec.impact?'<p class="ask__why" style="margin-top:5px;font-style:italic">'+esc(rec.impact)+"</p>":"")+
      (queue?'<div class="ask__queue">'+queue+"</div>":"")+
      "</div>";
  }

  /* ---- coverage ---- */
  var CHECK='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
  var HALF='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round"><line x1="5" y1="12" x2="19" y2="12"/></svg>';
  var RING='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"/></svg>';
  var COV_ICONS={confirmed:CHECK,partial:HALF,missing:RING};
  var COV_PILLS={confirmed:"covSeen",partial:"covPartial",missing:"covMissing"};
  function renderCoverage(){
    var s=state.session;
    if(!s||!s.coverage){
      el.coverage.innerHTML='<p class="empty-hint">'+esc(t("coverageEmpty"))+"</p>";
      el.coverageMeta.textContent=""; return;
    }
    var cats=s.coverage.categories;
    var seen=cats.filter(function(c){return c.status==="confirmed";}).length;
    el.coverageMeta.textContent=seen+" / "+cats.length;
    el.coverage.innerHTML=cats.map(function(c){
      return'<div class="cov cov--'+c.status+'">'+
        '<span class="cov__icon">'+(COV_ICONS[c.status]||RING)+"</span>"+
        '<span class="cov__label">'+esc(c.label)+"</span>"+
        '<span class="cov__pill">'+esc(t(COV_PILLS[c.status]))+"</span></div>";
    }).join("");
  }

  /* ---- comparables ---- */
  function renderComps(){
    var s=state.session,comps=s&&s.comparables;
    if(!comps||!comps.length){
      el.compsPanel.innerHTML='<div class="panel"><div class="panel__head"><span class="panel__title">'+
        esc(t("compsTitle"))+'</span></div><div class="panel__body"><p class="empty-hint">'+
        esc(t("compsEmpty"))+"</p></div></div>"; return;
    }
    var rows=comps.map(function(c){
      var sim=c.similarity||0;
      var fillColor=sim>80?"var(--grad-cool)":sim>60?"var(--grad-hot)":"linear-gradient(118deg,#f46060,#fb923c)";
      return"<tr>"+
        "<td><strong>"+(c.year||"—")+"</strong><br><span style='font-size:12px;color:var(--text-dim)'>"+(c.make||"")+" "+(c.model||"")+"</span></td>"+
        "<td class='num'>"+(c.mileage?Number(c.mileage).toLocaleString("en-US")+" mi":"—")+"</td>"+
        "<td class='num' style='font-weight:700'>$"+Number(c.price).toLocaleString("en-US")+"</td>"+
        "<td><div class='sim-bar'><div class='sim-bar__track'><div class='sim-bar__fill' style='width:"+sim+"%;background:"+fillColor+"'></div></div>"+
          "<span class='sim-bar__label'>"+sim+"%</span></div></td>"+
        "</tr>";
    }).join("");
    el.compsPanel.innerHTML='<div class="panel"><div class="panel__head">'+
      '<span class="panel__title">'+esc(t("compsTitle"))+'</span>'+
      '<span class="panel__meta">'+comps.length+" listings</span></div>"+
      '<div class="panel__body"><table class="comps-table">'+
        "<thead><tr><th>Vehicle</th><th>Mileage</th><th>Price</th><th>Similarity</th></tr></thead>"+
        "<tbody>"+rows+"</tbody></table>"+
        '<p class="method-note">'+esc(t("methodNote"))+"</p></div></div>";
  }

  /* ---- extracted features ---- */
  function renderFeatures(){
    var s=state.session,m=s&&s.merged_features;
    if(!m||!s.accepted_count){
      el.featuresPanel.innerHTML='<div class="panel"><div class="panel__head">'+
        '<span class="panel__title">'+esc(t("featTitle"))+'</span></div>'+
        '<div class="panel__body"><p class="empty-hint">'+esc(t("featEmpty"))+"</p></div></div>"; return;
    }
    function sevBars(val){
      var bars="";
      for(var i=1;i<=5;i++){
        var on=i<=val;
        var cls=on?(val>=4?"sev-bar__pip sev-bar__pip--on sev-bar__pip--severe":"sev-bar__pip sev-bar__pip--on"):"sev-bar__pip";
        bars+='<div class="'+cls+'"></div>';
      }
      return'<div class="sev-bar">'+bars+"</div>";
    }
    function featRow(label,value){
      if(value===null||value===undefined||value==="")return"";
      return'<div class="feat-item"><div class="feat-item__label">'+esc(label)+'</div>'+
        '<div class="feat-item__value">'+esc(value)+"</div></div>";
    }
    function sevRow(label,val){
      if(!val&&val!==0)return"";
      return'<div class="feat-item"><div class="feat-item__label">'+esc(label)+'</div>'+sevBars(val)+"</div>";
    }
    var mileage=m.mileage?Number(m.mileage).toLocaleString("en-US")+" "+(m.mileage_units||"mi"):null;
    var identity=m.identity_confidence?Math.round(m.identity_confidence*100)+"%":null;
    el.featuresPanel.innerHTML='<div class="panel"><div class="panel__head">'+
      '<span class="panel__title">'+esc(t("featTitle"))+'</span>'+
      '<span class="panel__meta">from vision model</span></div>'+
      '<div class="panel__body"><div class="feat-grid">'+
        featRow("Truck class",titleize(m.truck_class))+
        featRow("Make",m.make_guess)+
        featRow("Model",m.model_guess)+
        featRow("Year",m.year_guess)+
        featRow("Mileage",mileage)+
        featRow("Cab type",m.cab_type?titleize(m.cab_type):null)+
        featRow("Transmission",m.transmission_guess?titleize(m.transmission_guess):null)+
        featRow("Condition",m.overall_condition?titleize(m.overall_condition):null)+
        featRow("Identity certainty",identity)+
        sevRow("Rust severity",m.rust_severity)+
        sevRow("Body damage",m.body_damage_severity)+
        sevRow("Tire wear",m.tire_wear_severity)+
        sevRow("Interior wear",m.interior_wear_severity)+
        (m.frame_concern?'<div class="feat-item feat-item--wide"><div class="feat-item__value" style="color:var(--severe)">⚠ Frame concern flagged</div></div>':"")+
        (m.mechanical_warning?'<div class="feat-item feat-item--wide"><div class="feat-item__value" style="color:var(--severe)">⚠ Mechanical warning flagged</div></div>':"")+
      "</div></div></div>";
  }

  /* ---- specs ---- */
  function saveSpecs(){
    el.specsSave.disabled=true;
    api("PATCH","/api/sessions/"+state.sessionId+"/specs",{
      year:$("spec-year").value,make:$("spec-make").value,
      model:$("spec-model").value,trim:$("spec-trim").value,
      mileage:$("spec-mileage").value,mileage_units:$("spec-units").value,
    })
    .then(function(s){state.session=s;render();})
    .catch(function(err){flash(err.message);})
    .then(function(){el.specsSave.disabled=false;});
  }

  /* ---- theme ---- */
  function setTheme(theme){
    state.theme=theme;
    document.documentElement.setAttribute("data-theme",theme);
    $("theme-icon-moon").style.display=theme==="light"?"none":"block";
    $("theme-icon-sun").style.display=theme==="light"?"block":"none";
    try{localStorage.setItem("truckval-theme",theme);}catch(e){}
  }

  /* ---- boot ---- */
  function startSession(){
    return api("POST","/api/sessions").then(function(s){
      state.sessionId=s.session_id; state.session=s;
      state.thumbs={}; state.pending=[];
      setStatus("ready",s.mock_mode?"Offline demo":t("statusReady")); render();
    }).catch(function(err){setStatus("down","Server unreachable");flash(err.message||"Could not reach server.");});
  }

  function wire(){
    el.dropzone.addEventListener("dragover",function(e){e.preventDefault();el.dropzone.classList.add("is-over");});
    ["dragleave","dragend"].forEach(function(ev){
      el.dropzone.addEventListener(ev,function(){el.dropzone.classList.remove("is-over");});
    });
    el.dropzone.addEventListener("drop",function(e){
      e.preventDefault();el.dropzone.classList.remove("is-over");
      if(e.dataTransfer&&e.dataTransfer.files)enqueue(e.dataTransfer.files);
    });
    el.dropzone.addEventListener("click",function(e){
      if(e.target===el.pickBtn||el.pickBtn.contains(e.target))return; el.fileInput.click();
    });
    el.pickBtn.addEventListener("click",function(e){e.stopPropagation();el.fileInput.click();});
    el.fileInput.addEventListener("change",function(){enqueue(el.fileInput.files);el.fileInput.value="";});
    el.specsSave.addEventListener("click",saveSpecs);
    document.querySelectorAll(".specs input").forEach(function(inp){
      inp.addEventListener("keydown",function(e){if(e.key==="Enter")saveSpecs();});
    });
    el.resetBtn.addEventListener("click",function(){
      state.queue=[];
      ["spec-year","spec-make","spec-model","spec-trim","spec-mileage"].forEach(function(id){
        var e=$( id);if(e)e.value="";
      });
      startSession();
    });
    window.addEventListener("paste",function(e){
      if(e.clipboardData&&e.clipboardData.files&&e.clipboardData.files.length)enqueue(e.clipboardData.files);
    });
    $("theme-btn").addEventListener("click",function(){setTheme(state.theme==="dark"?"light":"dark");});
    $("lang-btn").addEventListener("click",function(){
      currentLang=currentLang==="en"?"tr":"en";
      $("lang-label").textContent=currentLang==="en"?"TR":"EN";
      applyI18n(); render();
    });
  }

  document.addEventListener("DOMContentLoaded",function(){
    el={
      dropzone:$("dropzone"),pickBtn:$("pick-btn"),fileInput:$("file-input"),
      photoList:$("photo-list"),verdict:$("verdict"),ask:$("ask"),
      coverage:$("coverage"),coverageMeta:$("coverage-meta"),
      compsPanel:$("comps-panel"),featuresPanel:$("features-panel"),
      alerts:$("alerts"),status:$("status"),statusText:$("status-text"),
      resetBtn:$("reset-btn"),specsSave:$("specs-save"),
    };
    try{var saved=localStorage.getItem("truckval-theme");if(saved)setTheme(saved);}catch(e){}
    wire(); applyI18n(); render(); startSession();
  });
})();
