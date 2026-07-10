let currentDevice=null;
async function loadDevice(){
 const serial=document.getElementById("serial").value.trim(),status=document.getElementById("status");
 if(!serial){status.textContent="Enter a serial number first.";return}
 status.textContent="Loading device...";
 try{
  const r=await fetch(`${API_BASE_URL}/device/${encodeURIComponent(serial)}`),d=await r.json();
  if(!r.ok)throw new Error(d.error||"Device not found");
  currentDevice=d;
  itemName.value=d.item_name||"";price.value=d.price||0;age.value=d.item_age_years||0;
  status.textContent="Device loaded.";
 }catch(e){status.textContent=e.message}
}
async function analyzeDevice(){
 const status=document.getElementById("status");
 status.textContent="Analyzing...";
 empty.classList.add("hidden");result.classList.remove("hidden");advice.textContent="Preparing concise AI advice...";
 try{
  const payload={item_name:itemName.value.trim()||"Unknown",price:Number(price.value||0),item_age_years:Number(age.value||0),condition:condition.value};
  const r=await fetch(`${API_BASE_URL}/predict`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
  const d=await r.json();if(!r.ok)throw new Error(d.error||"Prediction failed");
  decision.textContent=d.prediction||"—";currentValue.textContent=`$${Number(d.current_value||0).toFixed(2)}`;
  repairRatio.textContent=d.repair_cost_ratio??"—";category.textContent=d.item_category||"—";
  warranty.textContent=d.out_of_warranty?"Estimated expired":"Estimated active";
  await streamAdvice(payload,d);status.textContent="Analysis complete.";
 }catch(e){status.textContent=e.message;advice.textContent="Could not connect to the advisor."}
}
async function streamAdvice(device,prediction){
 const ar=getLang()==="ar";
 const response=await fetch(`${API_BASE_URL}/advisor/stream`,{method:"POST",headers:{"Content-Type":"application/json","Accept":"text/event-stream"},
 body:JSON.stringify({question:ar?"أعطني توصية قصيرة ومفيدة تشمل الضمان وأفضل إجراء والأجزاء القابلة للبيع والمواد القيّمة.":"Give a short useful recommendation including warranty, best action, sellable parts, and valuable materials.",serial_number:serial.value.trim()||null,language:ar?"ar":"en",device_context:{...device,...prediction,...(currentDevice||{})}})});
 if(!response.ok)throw new Error(`Advisor error ${response.status}`);
 const reader=response.body.getReader(),decoder=new TextDecoder();let buffer="",text="";advice.textContent="";
 while(true){const {value,done}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});
  const events=buffer.split("\n\n");buffer=events.pop()||"";
  for(const event of events){let type="",data="";for(const line of event.split("\n")){if(line.startsWith("event:"))type=line.slice(6).trim();if(line.startsWith("data:"))data+=line.slice(5).trim()}
   if(type==="delta"&&data){try{const o=JSON.parse(data);text+=o.text||"";advice.textContent=text}catch{}}
  }}
 if(!text)advice.textContent="No advice was returned.";
}
