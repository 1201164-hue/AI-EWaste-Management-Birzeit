function filterRows(){const q=search.value.toLowerCase();document.querySelectorAll("#rows tr").forEach(r=>r.style.display=r.textContent.toLowerCase().includes(q)?"":"none")}
