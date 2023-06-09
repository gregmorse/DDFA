//loading and parsing mnist file
var fs = require('fs');
var zlib = require('zlib');

//grab emitter (we're going to emit some events -- possibly happening in the browser later -- so we use component)
var emitter = (typeof process == "undefined" ? require('emitter') : require('component-emitter'));

module.exports = mnist;

function pullInt32(data, start)
{
	start = start || 0;
	//parse int 
	return ((data[start + 3] & 0xFF) << 0) |
	 ((data[start + 2] & 0xFF) << 8) |
	  ((data[start + 1] & 0xFF) << 16) |
	   ((data[start] & 0xFF) << 24);
}

function mnist()
{
  	var self = this;

  	//turn into an emitter
  	emitter(self);

  	self.desiredIgnoreMap = function(desiredArray)
  	{
		var inverseIgnore = {};

		for(var i=0; i < desiredArray.length; i++){
			var ix = parseInt(desiredArray[i]);
			inverseIgnore[ix] = true;
		}

		var toIgnore = {};
		for(var i=0; i < 10; i++)
		{
			if(!inverseIgnore[i])
				toIgnore[i] = true;
		}

		return toIgnore;
  	}

  	self.parseMNIST = function(labelPath, imagePath, ignore, chunkCount)
  	{
  		//if we send in a number, not ignore map
  		if(typeof ignore == "number")
  		{
  			chunkCount = ignore;
  			ignore = {};
  		}
  		else
  			ignore = ignore || {};

  		//we aggregate into a big image map
  		var imgMap = {};

  		//start parsing the labels
  		self.parseLabels(labelPath, function(err, labels)
  		{
  			//got an error? throw it please!
  			if(err)
  				throw err;

  			//otherwise, let's continue parsing the images now with our label mapping!
  			self.parseImages(imagePath, labels);
  		});

  		//when new image data comes to light, we include it in our map
  		self.on('parseFullImage', function(label, imgData)
  		{
  			//don't do anything with ignroed labels!
  			if(ignore[label])
  				return;

  			//labeles info
			var labeledImages = imgMap[label];

			//we haven't seen this label before, create a new array, and store it
			if(!labeledImages)
			{
				labeledImages = [];
				imgMap[label] = labeledImages;
			}

			//save the pixel info for this labeled object
			labeledImages.push(imgData);
  		});

  		self.on('parseEnd', function()
  		{
  			//report for now
  			console.log("Fully unzipped");

  			//now let's 
			var total = 0;

			//all the chunks we're going to break into
			var allPieces = [];

			//at least one chunk, dontcha think?
			var desiredPieceCount = chunkCount || 1;

			var labelToCount = {};

			for(var key in imgMap)
			{
				var cnt = imgMap[key].length;

				console.log("Digit '" + key + "' has " + cnt + " images");

				//we take the ceiling of the rounded -- that way, we are sure to catch everything in the future steps
				labelToCount[key] = {total: imgMap[key].length, current: 0, piece: Math.ceil(cnt/desiredPieceCount)};

				//how many total? should be the item count from before
				total += cnt;
			}

			//if you want it broken into pieces, we chunk it here
			if(desiredPieceCount > 1)
			{
				//break each one into desired piece chunks
				for(var i=0; i < desiredPieceCount; i++)
				{
					//desired chunk size == piece
					var piece = {};

					//loop through our big chunk of images -- slicing out the pieces we want 
					for(var digit in imgMap)
					{
						//grab current info
						var cntInfo = labelToCount[digit];

						//where do we start grabbing pictures?
						var start = cntInfo.current;

						//either we take a full chunk, or we take part of a chunk -- whichever is smaller
						var chunkLength = Math.min(cntInfo.piece, cntInfo.total - start);

						//slice out the images from our big array
						//grab images for this digit -- stick it in our chunk map
						piece[digit] = imgMap[digit].slice(start, start + chunkLength);

						//add the chunk length for next starting point!
						cntInfo.current+= chunkLength;

						console.log("Chunk of digit " + digit + ": ", piece[digit].length, " new current: ", cntInfo.current, "out of", cntInfo.total);

					}

					self.emit('parseDigitChunk', i, piece, desiredPieceCount);

					//all the pieces please
					allPieces.push(piece);
				}
			}

			//then send the full info 
			self.emit("parseFullData", imgMap);
  		})

  		
  	}

  	self.processImageData = function(pixelCount, data, startData, previousPixels)
  	{	
  		//either we start from previous images, or we start from scratch
		var currentImage = (previousPixels ? previousPixels : []);

		//all the full images we pick up!
		var fullImages = [];

		//start from start data inside our data chunk -- go to the end
		for(var i=startData; i < data.length; i++)
		{
			//just push pixels info into our image array as is
			currentImage.push(data[i]);

			//if you have at least some pixels, and number of pixels == pixelcount
			// we have a full image
			if(currentImage.length && (currentImage.length % pixelCount == 0))
			{
				//we have an image, save this full image
				fullImages.push(currentImage);
				//then reset our pixel holder
				currentImage = [];
			}	
		}

		//now we may have some images left over
		return {full: fullImages, partial: currentImage}
  	}

  	self.parseLabels = function(labelPath, finished)
  	{
  		var r = fs.createReadStream(labelPath);
		var z = zlib.createGunzip();

		var firstChunk = true;

		var labelMap = {};
		var labelCount = 0;
		var totalItemCount;

		//pipe our path for unzipping
		z.on('data', function(data) {

			//where to start reading the data
			var startData = 0;
			if(firstChunk)
			{
				//special case for first chunk start location
				startData = 8;

				var magicNumber = pullInt32(data);

				//check magicNumber
				if(magicNumber != 2049)
					throw new Error("Parse label error: magic number incorrect", magicNumber);

				//pull total count for verificationlater
				totalItemCount = pullInt32(data, 4);

				//has meta info about this chunk of data being parsed
				firstChunk = false;
			}

			//loop through each piece, storing label map
			for(var i=startData; i < data.length; i++)
			{
				//grab the label for this index
				labelMap[labelCount] = data[i];
				labelCount++;
			}
		}).on("end", function() {
			// response and decompression complete

			if(labelCount != totalItemCount)
				throw new Error("Incorrect item count :" + labelCount, " expecting: " + totalItemCount);

			//now let's send complete label mapping
			finished(undefined, labelMap);


		}).on("error", function(e) {

			//error!
			finished(e);

		});

		r.pipe(z);
  	}

  	self.parseImages = function(path, labelMap)
  	{
  		var r = fs.createReadStream(path);
		var z = zlib.createGunzip();

		var firstChunk = true;

		var images = [];

		var previousPixels;
		var totalPixelCount = 0;

		var imageCount = 0;

		//pipe our path for unzipping
		z.on('data', function(data) {

			var startData = 0;
			if(firstChunk)
			{
				//special case for first chunk start location
				startData = 16;

				var magicNumber = pullInt32(data);
				var itemCount = pullInt32(data, 4);
				var width = pullInt32(data, 8);
				var height = pullInt32(data, 12);

				//how many pixels in each image?  == width*height
				totalPixelCount = width*height;

				if(magicNumber != 2051)
					throw new Error("Expecting 2051, instead got image magic number: ", magicNumber);

				//has meta info about this chunk of data being parsed
				self.emit('parseMeta', itemCount, width, height);
				firstChunk = false;
			}

			//processing data for images
			//chunk data normally, passing in previous pixels object
			var processedImages = self.processImageData(totalPixelCount, data, startData, previousPixels);

			//all images processed from this data exist in processImages.full -- an array
			for(var i=0; i < processedImages.full.length; i++)
			{
				//emit this photo info!
				//in the future, this will come with a label
				self.emit('parseFullImage', labelMap[imageCount], processedImages.full[i]);

				//counting each image processed fully
				imageCount++;
			}

			//save the full images locally
			images = images.concat(processedImages.full);
			
			//then deal with the partial images from the data chunk
			previousPixels = processedImages.partial;

		}).on("end", function() {
			// response and decompression complete

			if(previousPixels.length)
				throw new Error("Preivous pixels left unprocessed: ", previousPixels.length);

			self.emit('parseEnd');

		}).on("error", function(e) {
			self.emit('parseError', e);
		});

		r.pipe(z);
  	}
  	

  	return self;
}

