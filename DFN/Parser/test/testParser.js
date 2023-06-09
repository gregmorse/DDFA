//pull in our parsers
var mnistParser = require("../lib/mnist-parser.js");
var path = require('path');
var fs = require('fs');

//get absolute path on system
var absDataPath = path.resolve(__dirname, "../data/t10k-images-idx3-ubyte.gz");
var absLabelPath = path.resolve(__dirname, "../data/t10k-labels-idx1-ubyte.gz");

//send for processing in parser
var parser = new mnistParser();
var chunkCount = 20;
function pullInt32(data, start)
{
	start = start || 0;
	//parse int 
	return ((data[start + 3] & 0xFF) << 0) |
	 ((data[start + 2] & 0xFF) << 8) |
	  ((data[start + 1] & 0xFF) << 16) |
	   ((data[start] & 0xFF) << 24);
}

//grab the meta info here 
parser.on('parseMeta', function(itemCount, width, height)
{
	console.log(" item count: ", itemCount, ", pixelWidth: ", width, ", pixelHeight: ", height);
});

parser.on('parseDigitChunk', function(chunkIx, chunkData, totalChunkCount)
{
	
	var save = JSON.stringify(chunkData);
	fs.writeFileSync(path.resolve(__dirname, '../../../data/chunks/' + chunkIx + ".json"), save);

	console.log("chunk-" + (chunkIx + 1) + " out of " + totalChunkCount + " images in total");
});

parser.on('parseFullData', function(imgMap){
	//if(chunkCount > 1)
	//	return;
		
	//otherwise, we simply save to chunk directory -- as the full.json object
	var save = JSON.stringify(imgMap);
	fs.writeFileSync(path.resolve(__dirname, '../../../data/chunks/full.json'), save);
});

parser.on('parseError', function(e)
{
	console.log("Parse error: ", e);
});

//ignore everything but 2 and 5
var ignore = parser.desiredIgnoreMap([2,5]);

parser.parseMNIST(absLabelPath, absDataPath, ignore, chunkCount);
