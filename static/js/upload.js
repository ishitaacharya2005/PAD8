// static/js/upload.js
document.addEventListener('DOMContentLoaded', function(){
  const form = document.getElementById('uploadForm');
  const progress = document.getElementById('uploadProgress');
  if (!form) return;

  form.addEventListener('submit', function(e){
    if (progress){
      progress.hidden = false;
      progress.value = 50;
      setTimeout(()=>{ progress.value = 85 }, 8000);
    }
  });
});
