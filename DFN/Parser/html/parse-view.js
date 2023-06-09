//need to load json info from disk
//also need to convert to base64 url image -- 2 simple tasks

var request = require('superagent');

var bitmapCreator = require('bitmap-creator');

module.exports = pview;

//the view from your moms house!
function pview()
{
	var self = this;

	//need a function to load sommmme data
	self.loadDigitData = function(location, finished)
	{
		request
			.get(location)
			.end(function(err, res)
			{
				if(err)
				{
					finished(err);
					return;
				}
				else if(res.statusCode == 404 || res.statusCode == 500)
				{
					finished(new Error("Not found or server error:" + res.statusCode));
				}
				else
				{
					//grab the response body -- this has our info
					var body = res.body;

					//parse the body string -  we need json
					// var jBody = JSON.parse(body);

					//send back the chunk body
					finished(undefined, body);
				}
			});
	}

	self.imageArrayToDataURL = function(array)
	{
		//need to convert the pixel info that we have
	 	// return bitmapCreator.bitmapeFromSinglePixelArray(imgArray);
	    var rows = Math.sqrt(array.length);
	    
	    var pixelRows= [];

	    var pixelCount =array.length-1;
	    for(var i=0; i < rows; i++)
	    {
	        var pRow = [];
	        for(var w=0; w < rows; w++){
	            //what is our single color?
	            var color = array[pixelCount - rows + w];

	            //apply colors to everyone
	            pRow.push([color,color,color]);
	        }

	        //then chunk to the next row (going backwards)
	        pixelCount -= rows;

	        //all done with that row
	        pixelRows.push(pRow);

	    }
	    //now we have pixel rows, send in that for generating bitmap
	    return bitmapCreator.generateBitmapDataURL(pixelRows, 1.0);
	}


	return self;
}













