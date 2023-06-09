using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Reflection;
using SharpNeat.Genomes.Neat;
using System.Xml;
using SharpNeat.Phenomes.NeuralNets;
using DFN.Evaluation;
using System.IO.Compression;

namespace DFN
{
    /// <summary>
    /// Save experimental data
    /// </summary>
    public class DFNSaver
    {
        static string experimentFolder;
        static string noveltyFolder;

        private static DFNSaver instance;

        //this is only ever called once -- this is a singleton
        private DFNSaver()
        {

        }

        public static DFNSaver Instance
        {
            get
            {
                if (instance == null)
                {
                    instance = new DFNSaver();
                }
                return instance;
            }
        }

        #region Methods [Access Folders]

        public string NoveltyFolderPath
        {
            get { return noveltyFolder; }
        }

        public string ExperimentFolderPath
        {
            get { return experimentFolder; }
        }

        #endregion

        #region Initialize Storage Directories

        public void createExperimentalDirectories()
        {
            string currentAssemblyDirectoryName = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            var now = DateTime.Now;
            int cnt = 0;
            string expName = "exp_" + now.Month + "_" + now.Day + "_" + now.Hour + "_" + now.Minute + "_" + cnt++;

            experimentFolder = currentAssemblyDirectoryName + "/" + expName;

            while (Directory.Exists(experimentFolder))
            {
                experimentFolder = experimentFolder.Substring(0, experimentFolder.LastIndexOf("_")) + "_" + cnt++;
            }

            //add the final touch
            experimentFolder += "/";
            noveltyFolder = experimentFolder + "novelty/";
            Directory.CreateDirectory(experimentFolder);

            //create novelty folder inside our experiment folder -- to store our archive
            Directory.CreateDirectory(noveltyFolder);
        }
        #endregion

        #region Saving SVM

        //public bool saveSVM(svm_model svm, string toSaveFolder, int featureCount, double testResults)
        //{
             
        //        System.Xml.Serialization.XmlSerializer writer =
        //            new System.Xml.Serialization.XmlSerializer(typeof(svm_model));

        //    string saveName= (toSaveFolder + "/" + string.Format("SVM_{0}_features_{1:0.000000}_performance.xml", featureCount, testResults));
        //    try 
        //    {
        //        System.IO.StreamWriter file = new System.IO.StreamWriter(saveName);
        //        //store the svm
        //        writer.Serialize(file, svm);
        //        file.Close();

        //    }
        //    catch(Exception e)
        //    {
        //        return false;
        //    }
            
        //    return true;
        //}

        #endregion

        #region Saving Perceptron

        static string aTesting = "testing";
        static string aTraining = "training";
        static string aGenomeID = "genomeID";
        static string aIx = "ix";
        static string aSource = "sourceIx";
        static string aTarget = "targetIx";
        static string aWeight = "weight";
        static string aDigitId = "digit";

        static string eNamePerceptron = "Perceptron";
        static string eNamePerformance = "Performance";
        static string eNameAllOutputs = "Outputs";
        static string eNameAllInputs = "Inputs";
        static string eNameInput = "input";
        static string eNameOutput = "output";

        static string eNameAllConnections = "Connections";
        static string eNameConn = "connection";

        public void savePerceptron(MultiPerceptron perceptron, double trainingAccuracy, double testAccuracy, IList<NeatGenome> archiveList, string overrideSaveFolder = null)
        {
            //start our XML doc
            XmlDocument doc = new XmlDocument();

            var perc = doc.CreateElement(eNamePerceptron);

            var perf = doc.CreateElement(eNamePerformance);
            perf.SetAttribute(aTraining, trainingAccuracy.ToString());
            perf.SetAttribute(aTesting, testAccuracy.ToString());
            perc.AppendChild(perf);

            var el = doc.CreateElement(eNameAllInputs);


            var inputCount = perceptron.InputCount;
            //let's pull apart this perceptron -- each index represents an index into the archive list
            for (int i = 0; i < inputCount; i++)
            {
                //this is how many inputs we have
                //each input is associated with a genomeID
                var genome = archiveList[i];
                if(genome != null)
                {
                    var iPut = doc.CreateElement(eNameInput);
                    
                    //this is the genome responsible for generating teh input for this guy
                    iPut.SetAttribute(aGenomeID, genome.Id.ToString());

                    //keep track of TRUE index of inputs -- includes bias node
                    iPut.SetAttribute(aIx, (i + 1).ToString());
                    el.AppendChild(iPut);
                }
            }
            //apend after fill
            perc.AppendChild(el);

            //now we need to do outputs (and their digit mappings)
            MNISTManager mm = MNISTManager.Instance;
            var interested = mm.InterestedDigits;
            string outputDigitString = "";

            //now we do outputs
            var oElement = doc.CreateElement(eNameAllOutputs);
           

            for (var i = 0; i < perceptron.OutputCount; i++)
            {
                outputDigitString += interested[i] + (i < perceptron.OutputCount - 1 ? "_" : "");

                var output = doc.CreateElement(eNameOutput);

                //this is the genome responsible for generating teh input for this guy
                output.SetAttribute(aDigitId, interested[i].ToString());

                //keep track of TRUE index of outputs -- includes bias node and input count
                output.SetAttribute(aIx, (i + inputCount + 1).ToString());

                oElement.AppendChild(output);

            }

            //apend after fill
            perc.AppendChild(oElement);

            var connections = doc.CreateElement(eNameAllConnections);

            int connCount = perceptron.ConnectionList.Length;
            for (int i = 0; i < connCount; i++)
            {
                var conn = perceptron.ConnectionList[i];

                var elConn = doc.CreateElement(eNameConn);
                elConn.SetAttribute(aSource, conn._srcNeuronIdx.ToString());
                elConn.SetAttribute(aTarget, conn._tgtNeuronIdx.ToString());
                elConn.SetAttribute(aWeight, conn._weight.ToString());
                elConn.SetAttribute(aIx, i.ToString());
                connections.AppendChild(elConn);
            }

            //apend after fill
            perc.AppendChild(connections);

            //all in one?
            doc.AppendChild(perc);


            // Save document
            string filename = (overrideSaveFolder == null ? ExperimentFolderPath : overrideSaveFolder) + "perceptron_" + perceptron.InputCount 
                + "_digits_" + outputDigitString + "_testPerf_" + String.Format("{0:0.000}", testAccuracy);

            filename += ".xml";

            //save our perceptron to file
            doc.Save(filename);
        }

        #endregion

        #region Behavior Saving

        static byte[] GetBytes(double[][] values)
        {
            int fullValueCount = 0;
            for(int i=0; i < values.Length; i++)
            {
                fullValueCount += values[i].Length;
            }

            var result = new byte[fullValueCount * sizeof(double)];
            fullValueCount = 0;
            for (int i = 0; i < values.Length; i++)
            {
                //how many doubles to copy in byte amounts?? Don't pass null objects or this will throw exception
                int len = values[i].Length * sizeof(double);
                
                //block copy, biatch!
                Buffer.BlockCopy(values[i], 0, result, fullValueCount, len);

                //now increment where we're copying inside our byte array
                fullValueCount += len;
            }

            return result;
        }
        static double[][] GetDoubles(byte[] bytes, int imageCount, int featureMinusPadding)
        {
            int featureCount = featureMinusPadding + 1;
            var result = new double[bytes.Length / sizeof(double)];
            Buffer.BlockCopy(bytes, 0, result, 0, bytes.Length);

            double[][] fullFeatures = new double[imageCount][];
            int srcIx = 0;
            for (int i = 0; i < imageCount; i++)
            {
                //create a new double array to hold this
                fullFeatures[i] = new double[featureCount];

                //make our array copy here, using the sources
                Array.Copy(result, srcIx, fullFeatures[i], 0, featureCount);

                //this is how many to skip to start the next copy operation
                srcIx += featureCount;
            }

            //send back the appropraite double[][] array, kill the super long double array
            //double memory size, but whatever
            return fullFeatures;
        }
        static string imageNamingString(string baseFolder, int featureCount, int startImageIx, int imageCount, int totalImages, int byteCount)
        {
            if (!Directory.Exists(baseFolder))
                Directory.CreateDirectory(baseFolder);

            return baseFolder + "/imageBehavior" 
                + "_start_" + startImageIx 
                + "_count_" + imageCount 
                + "_features_" +  featureCount 
                + "_total_" + totalImages 
                + "_bytes_" +byteCount
                + "_cache.gz"; 
        }

        /// <summary>
        /// Function to save image caches to file as a compressed byte array
        /// </summary>
        /// <param name="featureCache"></param>
        /// <param name="baseFolder"></param>
        /// <param name="genomeID"></param>
        /// <param name="featureCount"></param>
        /// <param name="startImageIx"></param>
        /// <param name="totalImages"></param>
        public void SaveImageCache(double[][] featureCache, string baseFolder, int featureCount, int startImageIx, int totalImages)
        {
            //we will now save the double array to file as a compressed binary json file
            byte[] bInfo = GetBytes(featureCache);

            int imageCount = featureCache.Length;

            string fileName = imageNamingString(baseFolder, featureCount, startImageIx, imageCount, totalImages, bInfo.Length);

            if (!File.Exists(fileName))
            {
                using (FileStream fileToCompress = File.Create(fileName))
                {
                    using (DeflateStream compressionStream = new DeflateStream(fileToCompress, CompressionMode.Compress))
                    {
                        //write our stream to file
                        compressionStream.Write(bInfo, 0, bInfo.Length);
                    }
                }
            }
        }
        public static string parseFileInfo(string fileName, string searchChunk)
        {
            int indexChunk = fileName.IndexOf(searchChunk + "_") + searchChunk.Length + 1;
            int nextUnderscore = indexChunk + fileName.Substring(indexChunk).IndexOf("_");
            return fileName.Substring(indexChunk, nextUnderscore - indexChunk);
        }

        /// <summary>
        /// Function to load image byte caches and restore them to their original size
        /// </summary>
        /// <param name="chunks"></param>
        /// <param name="fileName"></param>
        public void LoadImageCache(Dictionary<Tuple<int, int>, double[][]> chunks, string fileName)
        {
            //let's parse some info from the file name
            string byteChunk = parseFileInfo(fileName, "bytes");
            string imgChunk = parseFileInfo(fileName, "count");
            string featureChunk = parseFileInfo(fileName, "features");
            string imgStartChunk = parseFileInfo(fileName, "start");

            //the ints we need to be aware of
            int byteChunks;
            int imageCount;
            int featureCount;
            int imageStart;

            //need to parse all of these successfully
            if(int.TryParse(byteChunk, out byteChunks)
                && int.TryParse(imgChunk, out imageCount) 
                && int.TryParse(featureChunk, out featureCount)
                && int.TryParse(imgStartChunk, out imageStart))
            {
                byte[] decompressedBytes = new byte[byteChunks];
                using (FileStream fileToDecompress = File.Open(fileName, FileMode.Open))
                {
                    using (DeflateStream decompressionStream = new DeflateStream(fileToDecompress, CompressionMode.Decompress))
                    {
                        decompressionStream.Read(decompressedBytes, 0, byteChunks);

                        //now we have our bytes, let's make some doubles
                        double[][] features = GetDoubles(decompressedBytes, imageCount, featureCount);

                        //save our features, finish up the decompression
                        chunks.Add(new Tuple<int, int>(imageStart, imageStart + imageCount), features);
                    }
                }
            }
        }

        #endregion

        #region Population Saver

        string digitString()
        {
            //the genome with the highest novelty is sent in
            var outputs = DomainManager.Instance.getDigitsForString();

            string digitOutputString = "";
            for (int i = 0; i < outputs.Count; i++)
            {
                digitOutputString += outputs[i].ToString() + (i < outputs.Count - 1 ? "_" : "");
            }
            return digitOutputString;
        }

        public void savePopulationAndArchive(NeatGenome chmp, IList<NeatGenome> population, IList<NeatGenome> noveltyArchive, int genCount, string overrideSaveFolder = null)
        {
            //the genome with the highest novelty is sent in
            string digitOutputString = digitString();
            // Save document
            string filename = (overrideSaveFolder == null ? ExperimentFolderPath : overrideSaveFolder) + "pop_" + population.Count + "out_" + digitOutputString + "_gen_" + genCount + "_max_" + String.Format("{0:0.00}", chmp.EvaluationInfo.Fitness);
            filename += ".xml";

            //copy this list -- it may be modified in another thread while doing the save
            var pop = new List<NeatGenome>(population);


            //now emptys please
            if (pop != null && pop.Count > 0 && DFNLibrary.DFNParams.SAVEPOPULATION)
            {
                XmlDocument doc = NeatGenomeXmlIO.SaveComplete(pop, true);
                doc.Save(filename);
            }

            //now save novelty additions as well!
            saveNovelty(noveltyArchive, genCount, overrideSaveFolder);
        }

        HashSet<uint> savedArchiveGenomes = new HashSet<uint>();
        int archiveId = 0;
        public void saveNovelty(IList<NeatGenome> noveltyList, int genCount, string overrideSaveFolder = null)
        {
            //what didigts are we using?
            string digitOutputString = digitString();


            //cloen the list -- no threading issues while enumerating please
            List<NeatGenome> novArchive = new List<NeatGenome>(noveltyList);

            //what shall we save?
            List<NeatGenome> toSave = new List<NeatGenome>();
            
            //anythign that's not saved already!
            foreach (NeatGenome ng in novArchive)
            {
                //if we have your id, we're not going to save you again
                //however, if you are overriding the save location, then we assume you want the WHOLE archive saved so everythign is added when override != empty
                if (!savedArchiveGenomes.Contains(ng.Id) || overrideSaveFolder != null)
                {
                    toSave.Add(ng);
                }
            }

            //if we have some novel individuals to save
            if (toSave.Count > 0)
            {
                // Save document -- if we override the location, we just save there
                if(!Directory.Exists(overrideSaveFolder == null ? NoveltyFolderPath : overrideSaveFolder))
                    Directory.CreateDirectory(overrideSaveFolder == null ? NoveltyFolderPath : overrideSaveFolder);
                string filename = (overrideSaveFolder == null ? NoveltyFolderPath : overrideSaveFolder) + "novel_" + toSave.Count + "_out_" + digitOutputString + "_gen_" + genCount + "_nix_" + archiveId++;
                filename += ".xml";

                XmlDocument doc = NeatGenomeXmlIO.SaveComplete(toSave, true);
                doc.Save(filename);
            }


            //add ids of all saved!
            foreach (var ng in toSave)
                savedArchiveGenomes.Add(ng.Id);

        }

        
        #endregion

        #region Population Loader

        public List<NeatGenome> loadUniquePopulationInChunks(string folderName, out List<int> interestedDigits)
        {
            var allGenomes = loadPopulationIncrements(folderName, out interestedDigits);

            HashSet<uint> loadedGenomes = new HashSet<uint>();
            List<NeatGenome> rGenomes = new List<NeatGenome>();
            foreach (var genomes in allGenomes)
            {
                foreach (var g in genomes)
                {
                    if (!loadedGenomes.Contains(g.Id))
                    {
                        rGenomes.Add(g);
                        loadedGenomes.Add(g.Id);
                    }
                }
            }

            //send back everyone!
            return rGenomes;
        }

        /// <summary>
        /// TODO: Warning, load population assumes that the population loaded has the SAME number of inputs/outputs as the currently loaded MNIST database
        /// So if the function was saved with 0,2,5 digits, and then loaded with 3,4,5 digits, it's going to be very confused!
        /// </summary>
        /// <param name="folderName"></param>
        /// <returns></returns>
        public List<List<NeatGenome>> loadPopulationIncrements(string folderName, out List<int> interestedDigits)
        {
            //lets' pull in our population from this folder
            List<List<NeatGenome>> incrementalPopulations = new List<List<NeatGenome>>();

            //get a ll files
            var files = Directory.GetFiles(folderName);

            //remove anything that doesn't have the word "pop_" in it
            var filteredPops = files.Where(x=> x.Contains("novel_")).ToList();

            //sort in asciending order of file names!
            filteredPops.Sort();

            //just initialize all normal and stuff
            interestedDigits = new List<int>();

            //first load the original
            if (filteredPops.Count == 0)
                return null;

            HashSet<uint> popIds = new HashSet<uint>();
            //now let's pull ppopulations out of each object
            foreach(var popFileName in filteredPops)
            {
                XmlDocument pop = new XmlDocument();
                pop.Load(popFileName);

                if (popFileName.IndexOf("out_") != -1)
                {
                    var outLoc = popFileName.IndexOf("out_") + 4;
                    var genLoc = popFileName.IndexOf("_gen");
                    //loaded interested digits
                    var sub = popFileName.Substring(outLoc, genLoc- outLoc);

                    //somebody is non-plussed about this line '_'
                    var digits = sub.Split('_');
                    foreach (var d in digits)
                        if(!interestedDigits.Contains(int.Parse(d)))
                            interestedDigits.Add(int.Parse(d));

                }

                //don't need to send in a genome factory, just load the buggers
                var genomelist = NeatGenomeXmlIO.LoadCompleteGenomeList(pop, true);


                //we only want objects who we have never seen before
                List<NeatGenome> gList = new List<NeatGenome>();
                genomelist.ForEach(x=> {
                   if(!popIds.Contains(x.Id))
                   {
                       popIds.Add(x.Id);
                       gList.Add(x);
                   }
                });

                //only add objects we've never seen before -- this is just a precaution for no dupes
                incrementalPopulations.Add(gList);
            }

            return incrementalPopulations;
        }

        #endregion

    }
}
