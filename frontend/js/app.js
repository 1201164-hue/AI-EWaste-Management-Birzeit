const API_BASE_URL="https://ewaste-itad-api-hufud9ada8crcrhm.israelcentral-01.azurewebsites.net";
const dictionaries={
 en:{
  home:"Home",scan:"Scan & Analyze",database:"Database",analytics:"Analytics",agent:"AI Agent",about:"About Us",contact:"Contact",
  heroTitle:"Smarter Decisions",heroGreen:"Greener Tomorrow",
  heroText:"AI-powered ITAD system for smart, sustainable and responsible e-waste management.",
  scanNow:"Scan Device",openAgent:"Open AI Agent",liveImpact:"Live Impact",devices:"Devices Processed",
  co2:"CO₂ Saved (kg)",materials:"Materials Recovered (kg)",quickScan:"Scan & Analyze",
  quickScanText:"Analyze a device using serial number, age, price, and condition.",
  quickDatabase:"Device Database",quickDatabaseText:"Search and review university assets.",
  quickAnalytics:"Analytics",quickAnalyticsText:"View ITAD decisions and environmental impact.",
  quickAgent:"AI Advisor",quickAgentText:"Ask device-specific questions in real time."
 },
 ar:{
  home:"الرئيسية",scan:"المسح والتحليل",database:"قاعدة البيانات",analytics:"التحليلات",agent:"المستشار الذكي",about:"من نحن",contact:"اتصل بنا",
  heroTitle:"قرارات أذكى",heroGreen:"مستقبل أكثر خضرة",
  heroText:"نظام ذكي لإدارة أصول تكنولوجيا المعلومات والنفايات الإلكترونية بطريقة مستدامة ومسؤولة.",
  scanNow:"تحليل جهاز",openAgent:"فتح المستشار الذكي",liveImpact:"الأثر المباشر",devices:"الأجهزة المعالجة",
  co2:"ثاني أكسيد الكربون الموفر (كغم)",materials:"المواد المستردة (كغم)",quickScan:"المسح والتحليل",
  quickScanText:"حلّل الجهاز باستخدام الرقم التسلسلي والعمر والسعر والحالة.",
  quickDatabase:"قاعدة الأجهزة",quickDatabaseText:"ابحث واستعرض أصول الجامعة.",
  quickAnalytics:"التحليلات",quickAnalyticsText:"اعرض قرارات ITAD والأثر البيئي.",
  quickAgent:"المستشار الذكي",quickAgentText:"اطرح أسئلة خاصة بالجهاز واحصل على إجابة مباشرة."
 }
};
function getLang(){return localStorage.getItem("lang")||"en"}
function applyLanguage(){
 const lang=getLang(),d=dictionaries[lang];
 document.documentElement.lang=lang;
 document.documentElement.dir=lang==="ar"?"rtl":"ltr";
 document.body.dir=document.documentElement.dir;
 document.querySelectorAll("[data-i18n]").forEach(el=>{
   const key=el.dataset.i18n;
   if(d[key]!==undefined) el.textContent=d[key];
 });
 const langBtn=document.getElementById("langBtn");
 if(langBtn)langBtn.innerHTML=lang==="ar"?"<span>EN</span>":"<span>العربية</span>";
}
function toggleLanguage(){
 localStorage.setItem("lang",getLang()==="en"?"ar":"en");
 applyLanguage();
}
function applyTheme(){
 const theme=localStorage.getItem("theme")||"dark";
 document.body.classList.toggle("light",theme==="light");
 const b=document.getElementById("themeBtn");
 if(b)b.textContent=theme==="light"?"☾":"☀";
}
function toggleTheme(){
 localStorage.setItem("theme",(localStorage.getItem("theme")||"dark")==="dark"?"light":"dark");
 applyTheme();
}
function toggleMobile(){document.getElementById("mobileMenu")?.classList.toggle("open")}
function setActiveNav(){
 const page=document.body.dataset.page;
 document.querySelectorAll(`[data-page-link="${page}"]`).forEach(a=>a.classList.add("active"));
}
document.addEventListener("DOMContentLoaded",()=>{applyLanguage();applyTheme();setActiveNav()});
