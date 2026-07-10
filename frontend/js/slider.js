const heroImages=[
"https://images.unsplash.com/photo-1532601224476-15c79f2f7a51?auto=format&fit=crop&w=1800&q=85",
"https://images.unsplash.com/photo-1516321318423-f06f85e504b3?auto=format&fit=crop&w=1800&q=85",
"https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=1800&q=85",
"https://images.unsplash.com/photo-1611284446314-60a58ac0deb9?auto=format&fit=crop&w=1800&q=85"];
let heroIndex=0;
function renderHero(){
 document.getElementById("hero").style.backgroundImage=`url("${heroImages[heroIndex]}")`;
}
function nextHero(){
 heroIndex=(heroIndex+1)%heroImages.length;
 renderHero();
}
document.addEventListener("DOMContentLoaded",()=>{
 renderHero();
 setInterval(nextHero,6500);
});
