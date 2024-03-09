const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');

const inputSize = 28; // Size of the input image for your neural network
const outputSize = 280; // Size of the output canvas

let isDrawing = false;

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

function startDrawing(event) {
    isDrawing = true;
    draw(event);
}

function stopDrawing() {
    isDrawing = false;
}

// function draw(event) {
//     if (!isDrawing) return;
//     const x = event.offsetX;
//     const y = event.offsetY;
//     context.fillStyle = 'white';
//     context.fillRect(x, y, 15, 15); // Adjust size as needed
// }


function draw(event) {
    if (!isDrawing) return;
    const x = event.offsetX;
    const y = event.offsetY;

    // Calculate grayscale value for border (adjust as needed)
    const borderGrayValue = 225;

    // Set fill style for left border
    context.fillStyle = `rgb(${borderGrayValue}, ${borderGrayValue}, ${borderGrayValue})`;
    
    // Draw rectangle for left border
    context.fillRect(x - 7, y, 7, 7); // Adjust size as needed

    // Set fill style for right border
    context.fillStyle = `rgb(${borderGrayValue}, ${borderGrayValue}, ${borderGrayValue})`;
    
    // Draw rectangle for right border
    context.fillRect(x + 7, y, 7, 7); // Adjust size as needed

    // Set fill style for center (white)
    context.fillStyle = 'white';
    
    // Draw rectangle for center
    context.fillRect(x, y, 15, 15); // Adjust size as needed
}

function clearCanvas() {
    context.clearRect(0, 0, canvas.width, canvas.height);
}

function predict() {
    const largeCanvas = document.getElementById('canvas');
    const largeContext = largeCanvas.getContext('2d');
    const largeImageData = largeContext.getImageData(0, 0, 280, 280);

    const smallCanvas = document.createElement('canvas');
    const smallContext = smallCanvas.getContext('2d');
    smallCanvas.width = 28;
    smallCanvas.height = 28;

    // Draw the large canvas onto the small canvas, resizing it
    smallContext.drawImage(largeCanvas, 0, 0, 28, 28);

    // Get the image data from the small canvas
    const smallImageData = smallContext.getImageData(0, 0, 28, 28);

    // Convert image data to grayscale array
    const grayscaleArray = [];
    for (let i = 0; i < smallImageData.data.length; i += 4) {
        const grayscaleValue = (smallImageData.data[i] + smallImageData.data[i + 1] + smallImageData.data[i + 2]) / 3;
        grayscaleArray.push([grayscaleValue / 255]);
    }

    // Now you have a grayscale array of size 784 (28x28), which you can feed into your neural network for prediction
    console.log("Grayscale Array:", grayscaleArray);
    
    // Call your prediction function here and pass grayscaleArray as input
    sendPrediction(grayscaleArray);
}


function sendPrediction(pixelArray) {
    fetch('/predict', { 
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'pixelArray': pixelArray }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Prediction request failed');
        }
        return response.json();
    })
    .then(data => {
        console.log('Prediction data:', data);

        if (data.prediction === undefined) {
            throw new Error('Prediction result is undefined');
        }

        document.getElementById('prediction').innerHTML = `<h2 >Prediction: ${data.prediction}</h2>`;

        // Update bar chart with percentages
        updateBarChart(data.output);
    })
    .catch(error => {
        document.getElementById('prediction').innerHTML = `<h2>Prediction Error: ${error.message}</h2>`;
        console.error('Prediction error:', error);
    });
}


function updateBarChart(output) {
    // Get the canvas element
    var ctx = document.getElementById('barChart').getContext('2d');
    
    // Check if there is an existing chart instance
    if (window.myChart) {
        // Destroy the existing chart instance
        window.myChart.destroy();
    }
    
    // Create a new chart instance
    var newChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: output.map((_, index) => index),
            datasets: [{
                label: 'Percentage',
                data: output.map(value => parseFloat(value) * 100),
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Percentage (%)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Number'
                    }
                }
            }
        }
    });


    // Store the new chart instance in the window object
    window.myChart = newChart;
}


window.onload = function() {
    // Function to update the bar chart
    function updateBarChart(output) {
        // Get the canvas element
        var ctx = document.getElementById('barChart').getContext('2d');
        
        // Check if there is an existing chart instance
        if (window.myChart) {
            // Destroy the existing chart instance
            window.myChart.destroy();
        }
        
        // Create a new chart instance
        var newChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: output.map((_, index) => index),
                datasets: [{
                    label: 'Percentage',
                    data: output.map(value => parseFloat(value) * 100),
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Percentage (%)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Number'
                        }
                    }
                }
            }
        });

        // Store the new chart instance in the window object
        window.myChart = newChart;
    }
    
    // Call the function to update the bar chart with sample data (replace with your data)
    var sampleData = [0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9];
    updateBarChart(sampleData);
}
