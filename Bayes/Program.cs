using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Bayes
{
    class Program
    {
        static void Main (string[] args)
        {
            string TrainingDataPath = args[0];
            string TestingDataPath = args[1];
            double classPriorDelta = Convert.ToDouble(args[2]);
            double condProbDelta = Convert.ToDouble(args[3]);
            int Docid = 0;
            Dictionary<string, bool> Vocab = new Dictionary<string, bool>();
            //read docs
            Dictionary<string, List<Document>> TrainingDocs = ReadData(TrainingDataPath, ref Docid,true,ref Vocab);
            double totalTrainingDocs = Docid;
            double DocsPerClass = 0;
            string featureClass;
            double prob = 0;
            Dictionary<string, double> classPrior = new Dictionary<string, double>();
            Dictionary<string, double> FeatureClassProb = new Dictionary<string, double>();
            int totalClass =TrainingDocs.Count;
            //calculating Priors
            foreach (var classLabel in TrainingDocs)
            {
                DocsPerClass = classLabel.Value.Count;
                //calculate Prior
                prob = System.Math.Log10((DocsPerClass + classPriorDelta) / (totalTrainingDocs + (classPriorDelta * totalClass)));
                classPrior.Add(classLabel.Key, prob);
                foreach (var word in Vocab)
                {
                    featureClass = classLabel.Key + "_" + word.Key;
                    int wordcount = 0;
                    foreach (var doc in classLabel.Value)
                    {
                        if (doc.WordCount.ContainsKey(word.Key))
                            wordcount++;
                    }
                    //Calculate prob of each word given a class 
                    prob = System.Math.Log10((wordcount + condProbDelta) / (DocsPerClass + (condProbDelta*2)));
                    FeatureClassProb.Add(featureClass, prob);
                }
                
            }
            //Classifying training Docs

            Dictionary<String, int> ConfusionDict = new Dictionary<string, int>();
            var UniqueClasses = TrainingDocs.Keys.ToList();
            Dictionary<int, List<ProbClass>> DocClassProb = new Dictionary<int, List<ProbClass>>();
            string ConfusionDictKey;
            foreach (var classGroup in TrainingDocs)
            {
                
                foreach (var document in classGroup.Value)
                {
                    ProbClass maxProbClass = new ProbClass();
                    foreach (var genClass in UniqueClasses)
                    {
                        double Docprob = 0;
                        
                        foreach (var word in Vocab)
                        {
                            featureClass = genClass+"_"+word.Key;
                            if (document.WordCount.ContainsKey(word.Key))
                                Docprob += FeatureClassProb[featureClass];
                            else
                                Docprob += (1-FeatureClassProb[featureClass]);
                        }
                        Docprob += classPrior[genClass];
                        if(Docprob > maxProbClass.prob)
                        {
                            maxProbClass.classLabel = genClass;
                            maxProbClass.prob = Docprob;
                        }
                        ProbClass temp = new ProbClass(genClass, Docprob);
                        if (DocClassProb.ContainsKey(document.Doc_id))
                            DocClassProb[document.Doc_id].Add(temp);
                        else
                            DocClassProb.Add(document.Doc_id, new List<ProbClass>() { temp });
                    }
                    ConfusionDictKey = classGroup.Key + "_" + maxProbClass.classLabel;
                    if (ConfusionDict.ContainsKey(ConfusionDictKey))
                        ConfusionDict[ConfusionDictKey]++;
                    else
                        ConfusionDict.Add(ConfusionDictKey, 1);
                }
            }



            Dictionary<string, List<Document>> TestingDocs = ReadData(TestingDataPath, ref Docid,false,ref Vocab);





            Console.WriteLine("done");
            Console.ReadLine();
        }

        public static Dictionary<string, List<Document>> ReadData (string TrainingDataPath, ref int Docid, bool training, ref Dictionary<string, bool> Vocab)
        {
            string line, key;
            int value, index;
            Dictionary<string, List<Document>> TrainingClassDict = new Dictionary<string, List<Document>>();

	        using(StreamReader Sr = new StreamReader(TrainingDataPath))
		        {
			        while((line = Sr.ReadLine())!=null)
			        {
				        if (String.IsNullOrWhiteSpace(line))
					        continue;
				        string[] words = line.Split(new string[] { " " }, StringSplitOptions.RemoveEmptyEntries);
				        Document temp = new Document(Docid++);
				        for (int i = 1; i < words.Length; i++)
				        {
					        index = words[i].IndexOf(":");
					        key = words[i].Substring(0,index );
					        value = Convert.ToInt32(words[i].Substring(index + 1));
					        if (temp.WordCount.ContainsKey(key))
						        temp.WordCount[key] += value;
					        else
						        temp.WordCount.Add(key, value);
                            if(training)
                            {
                                if (!Vocab.ContainsKey(key))
                                    Vocab.Add(key, true);
                            }
				        }
                        if (TrainingClassDict.ContainsKey(words[0]))
                            TrainingClassDict[words[0]].Add(temp);
                        else
                            TrainingClassDict.Add(words[0], new List<Document>() { temp });
                            
			        }
		        }
            return TrainingClassDict;
        }
    }
}
