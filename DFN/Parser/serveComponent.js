var express = require('express');

var app = express();

app.use("/data", express.static( __dirname + "/build/chunks"));
// app.use("/", express.static( __dirname + "/html"));
app.use(express.static(__dirname + "/html"));

var rPort = Math.floor(7000);

app.listen(rPort, function()
{
	console.log('Listening on port: ' + rPort);
});


