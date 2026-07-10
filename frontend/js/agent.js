let selectedDevice=null;
let activeStreamBubble=null;

async function loadAgentDevice(){
  const s=document.getElementById("serial").value.trim();
  const status=document.getElementById("agentStatus");

  if(!s){
    status.textContent="Enter a serial number first.";
    return;
  }

  status.textContent="Loading device...";

  try{
    const r=await fetch(`${API_BASE_URL}/device/${encodeURIComponent(s)}`);
    const d=await r.json();

    if(!r.ok){
      throw new Error(d.error||"Device not found");
    }

    selectedDevice=d;
    document.getElementById("deviceStatus").textContent=`${d.item_name||"Device"} • ${d.serial_number||s}`;
    const meta=document.getElementById("deviceMeta");
    if(meta) meta.textContent=`${d.item_category||"Unknown category"} • ${d.status||"Unknown status"}`;
    status.textContent="Device loaded successfully.";
  }catch(e){
    status.textContent=e.message||"Could not load device.";
  }
}

function usePrompt(text){
  const q=document.getElementById("question");
  q.value=text;
  q.focus();
  autoResizeComposer();
}

function handleComposerKey(event){
  if(event.key==="Enter"&&!event.shiftKey){
    event.preventDefault();
    askAgent();
  }
  requestAnimationFrame(autoResizeComposer);
}

function autoResizeComposer(){
  const q=document.getElementById("question");
  q.style.height="auto";
  q.style.height=Math.min(q.scrollHeight,110)+"px";
}

function clearChat(){
  document.getElementById("chat").innerHTML=`
    <div class="chat-message assistant">
      <div class="message-avatar">AI</div>
      <div class="message-bubble">
        Ask me about the device condition, warranty, repair, resale value, reusable parts, recyclable materials, or the recommended ITAD decision.
      </div>
    </div>`;
  document.getElementById("agentStatus").textContent="";
}

async function askAgent(){
  const questionInput=document.getElementById("question");
  const q=questionInput.value.trim();
  if(!q)return;

  appendMessage("user",q);
  questionInput.value="";
  autoResizeComposer();

  const assistantBubble=appendMessage("assistant","");
  activeStreamBubble=assistantBubble;
  assistantBubble.classList.add("typing");
  assistantBubble.textContent="Thinking...";

  const status=document.getElementById("agentStatus");
  status.textContent="Generating response...";

  try{
    const r=await fetch(`${API_BASE_URL}/advisor/stream`,{
      method:"POST",
      headers:{
        "Content-Type":"application/json",
        "Accept":"text/event-stream"
      },
      body:JSON.stringify({
        question:q,
        serial_number:document.getElementById("serial").value.trim()||null,
        language:getLang()==="ar"?"ar":"en",
        device_context:selectedDevice||null
      })
    });

    if(!r.ok){
      throw new Error(`Advisor error ${r.status}`);
    }

    const reader=r.body.getReader();
    const decoder=new TextDecoder();
    let buffer="";
    let text="";

    assistantBubble.textContent="";
    assistantBubble.classList.remove("typing");

    while(true){
      const {value,done}=await reader.read();
      if(done)break;

      buffer+=decoder.decode(value,{stream:true});
      const events=buffer.split("

");
      buffer=events.pop()||"";

      for(const event of events){
        let type="";
        let data="";

        for(const line of event.split("
")){
          if(line.startsWith("event:")){
            type=line.slice(6).trim();
          }
          if(line.startsWith("data:")){
            data+=line.slice(5).trim();
          }
        }

        if(type==="delta"&&data){
          try{
            const obj=JSON.parse(data);
            text+=obj.text||"";
            assistantBubble.textContent=text;
            scrollChat();
          }catch{}
        }
      }
    }

    if(!text){
      assistantBubble.textContent="No response was returned.";
    }

    status.textContent="";
  }catch(e){
    assistantBubble.classList.remove("typing");
    assistantBubble.textContent="Could not connect to the advisor right now.";
    status.textContent=e.message||"Advisor connection failed.";
  }
}

function appendMessage(role,text){
  const chat=document.getElementById("chat");
  const wrapper=document.createElement("div");
  wrapper.className=`chat-message ${role}`;

  const avatar=document.createElement("div");
  avatar.className="message-avatar";
  avatar.textContent=role==="assistant"?"AI":"You";

  const bubble=document.createElement("div");
  bubble.className="message-bubble";
  bubble.textContent=text;

  if(role==="user"){
    wrapper.appendChild(bubble);
    wrapper.appendChild(avatar);
  }else{
    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);
  }

  chat.appendChild(wrapper);
  scrollChat();
  return bubble;
}

function scrollChat(){
  const chat=document.getElementById("chat");
  chat.scrollTop=chat.scrollHeight;
}

document.addEventListener("DOMContentLoaded",()=>{
  document.getElementById("question")?.addEventListener("input",autoResizeComposer);
});
