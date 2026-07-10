function toggleMiniChat(){
  const panel=document.getElementById("miniChatPanel");
  if(!panel)return;
  const open=panel.classList.toggle("open");
  panel.setAttribute("aria-hidden",open?"false":"true");
  if(open)setTimeout(()=>document.getElementById("miniQuestion")?.focus(),120);
}

async function sendMiniChat(){
  const input=document.getElementById("miniQuestion");
  const serial=document.getElementById("miniSerial");
  const messages=document.getElementById("miniChatMessages");
  if(!input||!messages)return;

  const q=input.value.trim();
  if(!q)return;

  const user=document.createElement("div");
  user.className="mini-msg user";
  user.textContent=q;
  messages.appendChild(user);

  const ai=document.createElement("div");
  ai.className="mini-msg ai";
  ai.textContent="Thinking...";
  messages.appendChild(ai);

  input.value="";
  messages.scrollTop=messages.scrollHeight;

  try{
    const r=await fetch(`${API_BASE_URL}/advisor/stream`,{
      method:"POST",
      headers:{"Content-Type":"application/json","Accept":"text/event-stream"},
      body:JSON.stringify({
        question:q,
        serial_number:serial?.value.trim()||null,
        language:getLang()==="ar"?"ar":"en"
      })
    });

    if(!r.ok)throw new Error(`Advisor error ${r.status}`);

    const reader=r.body.getReader();
    const decoder=new TextDecoder();
    let buffer="",text="";
    ai.textContent="";

    while(true){
      const {value,done}=await reader.read();
      if(done)break;
      buffer+=decoder.decode(value,{stream:true});
      const events=buffer.split("\n\n");
      buffer=events.pop()||"";

      for(const event of events){
        let type="",data="";
        for(const line of event.split("\n")){
          if(line.startsWith("event:"))type=line.slice(6).trim();
          if(line.startsWith("data:"))data+=line.slice(5).trim();
        }
        if(type==="delta"&&data){
          try{
            const obj=JSON.parse(data);
            text+=obj.text||"";
            ai.textContent=text;
            messages.scrollTop=messages.scrollHeight;
          }catch{}
        }
      }
    }

    if(!text)ai.textContent="No response was returned.";
  }catch(e){
    ai.textContent="Could not connect to the advisor.";
  }
}
