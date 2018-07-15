const app = require('express')();
const http = require('http').Server(app);
const io = require('socket.io')(http);
const fs = require('fs');
const tf = require('@tensorflow/tfjs');

// Load the binding:
require('@tensorflow/tfjs-node');

var model;
tf.loadModel('file://../model.tfjs/model.json').then(x => {
  console.log('Model loaded');
  model = x;
})

app.get('/', function(req, res){
  res.sendFile(__dirname + '/index.html');
});

var data = {};
var states = {};
var selectedNames = [];
var validatedMoves = {};

var controlSocket;
var connectedDevices = [];
var evaluate = false;
var evaluating = false;

const MOVES = ['tob', 'pae', 'charge', 'attack', 'shield']

function applyFilter(name, newMove) {
  if (!states[name] || states[name].lastMove != newMove) {
    states[name] = {
      lastMove: newMove,
      lastTime: Date.now()
    };
    return 0;
  } else {
    var time = Date.now() - states[name].lastTime;
    if (time > 100) {
      return time;
    } else {
      return 0;
    }
  }
}

function parseData(arr) {
  // Same logic as in python
  var prevAccel = null;
  var prevGyro = null;
  var prevType = '';
  var data = [];
  
  var i = 0;
  arr.forEach(current => {
    if (current.type == 'gyro') prevGyro = current;
    if (prevType == 'accel' && prevGyro) {
      i++;
      if (i % 2 == 1)
        data.push([prevAccel.x, prevAccel.y, prevAccel.z, prevGyro.x, prevGyro.y, prevGyro.z]);
    }
    prevType = current.type;
    if (current.type == 'accel') prevAccel = current;
  });

  return data;
}

function clean(str) {
  return str.toLowerCase().replace(/[^a-zA-Z0-9]+/g, "-");
}

function updateDevices() {
  if (controlSocket) {
    controlSocket.emit('devices', connectedDevices.map(x => x.name));
  }
}

function argMax(arr) {
  var max = 0;
  for (var i = 0; i < arr.length; i++) {
    if (arr[i] > arr[max]) {
      max = i;
    }
  }
  return max;
}

io.on('connection', function(socket){
  socket.on('config', function(msg) {
    if (msg.name) {
      console.log('a user connected: ' + msg.name);
      socket.name = msg.name;
      connectedDevices.push(socket);
    } else if (msg.control) {
      console.log('controller has connected');
      controlSocket = socket;
    }
    updateDevices();
  });
  socket.on('disconnect', function() {
    if (socket.name) {
      console.log('disconnected: ' + socket.name);
    } else if (controlSocket == socket) {
      console.log('controller has disconnected');
      controlSocket = null;
    }
    var index = connectedDevices.indexOf(socket);
    if (index !== -1) connectedDevices.splice(index, 1);

    updateDevices();
  })
  socket.on('data', function(msg) {
    if (selectedNames.indexOf(socket.name) != -1) {
      if (!data[socket.name]) data[socket.name] = [];
      var thisData = data[socket.name];
      thisData.push(msg);
      
      if (evaluate && controlSocket && !evaluating && thisData.length > 50) {
        evaluating = true
        const start = Date.now()
        const parsedData = [parseData(thisData.slice(-100))];
        var result = model.predict(tf.tensor3d(parsedData));
        result.data().then(data => {
          evaluating = false
          const max = argMax(data);
          if (data[max] >= 0.5) {
            const time = Date.now() - start;
            const ms = applyFilter(socket.name, MOVES[max]);
            if (ms) {
              validatedMoves[socket.name] = MOVES[max];
            }
            // console.log("Evaluation " + socket.name + ": <font size=20>" + MOVES[max] + "</font> with stable time " + ms + " ms in " + time + " ms");
            controlSocket.emit("evaluation", {msg: "Evaluation " + socket.name + ": <font size=20>" + validatedMoves[socket.name] + "</font> (" + MOVES[max] +") with stable time " + ms + " ms in " + time + " ms",
              device: clean(socket.name)});
          }
        });
      }
      // Periodically clear
      if (thisData.length > 5000) {
        data[socket.name] = thisData.slice(-500);
      }
    }
  });
  socket.on('control', function(msg) {
    if (msg.act == "save") {
      selectedNames.forEach(selectedName => {
        var filename = 'data/' + clean(msg.type) + '/' + clean(selectedName) + '_' + Date.now() + '.json';
        fs.writeFileSync(filename, data[selectedName].slice(-200).map(x => JSON.stringify(x)).join('\n'));
        controlSocket.emit("status", "Saved to " + filename);
      })
    }
    if (msg.act == "device") {
      selectedNames = msg.names;
      controlSocket.emit("status", "Set current device to " + selectedNames);
    }
    if (msg.act == "evaluate") {
      evaluate = !evaluate;
    }
  });
});

http.listen(3000, function(){
  console.log('listening on *:3000');
});

