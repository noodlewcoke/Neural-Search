onst canvas = document.querySelector('#canvas');
const ctx = canvas.getContext('2d');

canvas.width = 400;
canvas.height = 400;
ctx.strokeStyle = 'white';
ctx.lineJoin = 'round'; 
ctx.lineCap = 'round';
ctx.lineWidth = 40;
ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height)


let drawingMode = false;
let cX = 0; //X coordinate
let cY = 0; //Y coordinate
let changingColor = 0; //color variable responsible for changing color
let direction = true; //changes the size of the brush

function draw(e) {
    if (!drawingMode) return;
    // ctx.strokeStyle = `hsl(${changingColor}, 100%, 50%)`;
    ctx.beginPath();
    ctx.moveTo(cX, cY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    [cX, cY] = [e.offsetX, e.offsetY];
    
    if (changingColor >= 360) changingColor = 0;
    
    
    if (ctx.lineWidth >= 100 || ctx.lineWidth <= 1) {
        direction = !direction; //if true then it goes false, same goes for false
    }
    
}

canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mousedown', (e) => {
    drawingMode = true;
    [cX, cY] = [e.offsetX, e.offsetY]; //on mousedown initiate coordinates for X and Y
});
canvas.addEventListener('mouseup', () => drawingMode = false);
canvas.addEventListener('mouseout', () => drawingMode = false);



var elem = document.getElementById('clear-button')
elem.onclick= function(){
    ctx.strokeStyle = 'white';
    ctx.lineJoin = 'round'; 
    ctx.lineCap = 'round';
    ctx.lineWidth = 40;
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    }

var search_button = document.getElementById("search-button")
search_button.onclick = function(){
var img = canvas.toDataURL('image/jpg');


var xhr = new XMLHttpRequest();
xhr.open("POST", "http://127.0.0.1:5000/predict", true);
xhr.setRequestHeader("Content-Type", "application/json");
xhr.send(JSON.stringify({
    'file': img
}));


xhr.onload = function () {
    if (xhr.status >= 200 && xhr.status < 300) {
        const response = JSON.parse(xhr.responseText);
        for (let i=0; i < response.top_25.length; i++) {
            var delImage = document.getElementById('test'+i);
            if (delImage != null) {
                delImage.parentNode.removeChild(delImage);
            } 
            image = response.top_25[i][1];
            var test_image = document.createElement('img');
            test_image.src = "data:image/png;base64," + image;
            test_image.width = 64;
            test_image.height = 64;
            test_image.id = 'test'+ i;
            document.body.appendChild(test_image); 
        }

    }
};


}