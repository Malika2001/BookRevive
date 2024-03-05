function toggleCaptcha() {
    var captchaContainer = document.getElementById('captchaContainer');
    if (document.getElementById('_checkbox').checked) {
      captchaContainer.style.display = 'block';
    } else {
      captchaContainer.style.display = 'none';
    }
  }
  
document.getElementById('showModalBtn').addEventListener('click', function() {
    console.log(document.getElementById('Modal'))
    document.getElementById('Modal').style.display = 'block';
});

// Close the modal when clicking outside the modal content
window.addEventListener('click', function(event) {
    var modal = document.getElementById('Modal');
    if (event.target == modal) {
        modal.style.display = 'none';
    }
});

  