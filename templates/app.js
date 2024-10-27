document.getElementById('inputMethod').addEventListener('change', function() {
    const method = this.value;
    document.getElementById('pdfUpload').style.display = method === 'pdf' ? 'block' : 'none';
    document.getElementById('imageUpload').style.display = method === 'image' ? 'block' : 'none';
    document.getElementById('manualInput').style.display = method === 'manual' ? 'block' : 'none';
}); 

function processInput(method) {
    const form = document.getElementById('upload Form');
    const formData = new FormData(form);
    formData.append('inputMethod', method);

    fetch('/process', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('result').innerHTML = `<p class="error">${data.error}</p>`;
        } else {
            document.getElementById('result').innerHTML = `<p>${data.result}</p>`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerHTML = '<p class="error">An error occurred while processing your request.</p>';
    });
}