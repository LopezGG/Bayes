using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MultinomialBayes
{
    class Program
    {
        static void Main (string[] args)
        {
            string TrainingDataPath = args[0];
            string TestingDataPath = args[1];
            double classPriorLogDelta = Convert.ToDouble(args[2]);
            double condProbDelta = Convert.ToDouble(args[3]);
            string modelFile = args[4];
            string sysOutput = args[5];
            if (File.Exists(modelFile))
            {
                File.Delete(modelFile);
            }
            if (File.Exists(sysOutput))
            {
                File.Delete(sysOutput);
            }
            int Docid = 0;
            Dictionary<string, bool> Vocab = new Dictionary<string, bool>();
            //read docs
            Dictionary<string, List<Document>> TrainingDocs = ReadData(TrainingDataPath, ref Docid, true, ref Vocab);
            double totalTrainingDocs = Docid;
            double DocsPerClass = 0;
            string featureClass;
            double prob = 0;
            Dictionary<string, double> classPriorLog = new Dictionary<string, double>();
            Dictionary<string, double> classPrior = new Dictionary<string, double>();
            Dictionary<string, double> FeatureClassProbLog = new Dictionary<string, double>();
            Dictionary<string, double> FeatureClassProb = new Dictionary<string, double>();
            int totalClass = TrainingDocs.Count;
            //calculating Priors
            foreach (var classLabel in TrainingDocs)
            {
                DocsPerClass = classLabel.Value.Count;
                //calculate Prior
                prob = ((DocsPerClass + classPriorLogDelta) / (totalTrainingDocs + (classPriorLogDelta * totalClass)));
                classPriorLog.Add(classLabel.Key, System.Math.Log10(prob));
                classPrior.Add(classLabel.Key, prob);
                int totalWordsPerClass = 0;
                //get total word count of all words in doc to calculate prob
                foreach (var word in Vocab)
                {
                    foreach (var doc in classLabel.Value)
                    {
                        if (doc.WordCount.ContainsKey(word.Key))
                            totalWordsPerClass += doc.WordCount[word.Key];
                    }
                }

                foreach (var word in Vocab)
                {
                    featureClass = classLabel.Key + "_" + word.Key;
                    int wordcount = 0;
                    foreach (var doc in classLabel.Value)
                    {
                        if (doc.WordCount.ContainsKey(word.Key))
                            wordcount+=doc.WordCount[word.Key];
                    }
                    //Calculate prob of each word given a class 
                    prob = (wordcount + condProbDelta) / (totalWordsPerClass + (condProbDelta * Vocab.Count));
                    double logProb = System.Math.Log10(prob);
                    FeatureClassProbLog.Add(featureClass, logProb);
                    FeatureClassProb.Add(featureClass, prob);

                }

            }

            string classlabel, featureValue;
            using (StreamWriter Sw = new StreamWriter(modelFile))
            {
                Sw.WriteLine("%%%%% prior prob P(c) %%%%%");
                foreach (var classGroup in classPriorLog)
                {
                    Sw.WriteLine(classGroup.Key + "\t" + classPrior[classGroup.Key] + "\t" + classGroup.Value);
                }
                Sw.WriteLine("%%%%% conditional prob P(f|c) %%%%%");
                Sw.WriteLine("%%%%% conditional prob P(f|c) c=talk.politics.guns %%%%%");

                foreach (var feature in FeatureClassProbLog)
                {
                    classlabel = feature.Key.Substring(0, feature.Key.IndexOf("_"));
                    featureValue = feature.Key.Substring(feature.Key.IndexOf("_") + 1);
                    Sw.WriteLine(featureValue + "\t" + classlabel + "\t" + FeatureClassProb[feature.Key] + "\t" + feature.Value);
                }
            }



            ClassifyandWrite(sysOutput, TrainingDocs, Vocab, FeatureClassProbLog, classPriorLog, "train");
            Dictionary<string, List<Document>> TestingDocs = ReadData(TestingDataPath, ref Docid, false, ref Vocab);
            ClassifyandWrite(sysOutput, TestingDocs, Vocab, FeatureClassProbLog, classPriorLog, "test");
            //Console.WriteLine("Done");
            //Console.ReadLine();
        }
        public static void ClassifyandWrite (string sysOutput, Dictionary<string, List<Document>> TrainingDocs, Dictionary<string, bool> Vocab, Dictionary<string, double> FeatureClassProbLog, Dictionary<string, double> classPriorLog, string testOrTrain)
        {
            StreamWriter Sw1 = new StreamWriter(sysOutput,true);
            string featureClass;
            Sw1.WriteLine("%%%%% " + testOrTrain + " data:");
            Dictionary<String, int> ConfusionDict = new Dictionary<string, int>();
            string st1, st2;
            var UniqueClasses = TrainingDocs.Keys.ToList();
            string ConfusionDictKey;
            double totalDoc = 0;
            foreach (var classGroup in TrainingDocs)
            {

                foreach (var document in classGroup.Value)
                {
                    
                    Dictionary<string, double> ClassProbDoc = new Dictionary<string, double>();
                    foreach (var genClass in UniqueClasses)
                    {
                        double Docprob = 0;

                        foreach (var word in Vocab)
                        {
                            featureClass = genClass + "_" + word.Key;
                            if (document.WordCount.ContainsKey(word.Key))
                                Docprob += (document.WordCount[word.Key] * FeatureClassProbLog[featureClass]);
                        }
                        Docprob += classPriorLog[genClass];
                        ClassProbDoc.Add(genClass, Docprob);
                    }
                    var orderedDict = ClassProbDoc.OrderByDescending(x => x.Value).ToDictionary(x => x.Key, x => x.Value);
                    ConfusionDictKey = classGroup.Key + "_" + orderedDict.First().Key;
                    if (ConfusionDict.ContainsKey(ConfusionDictKey))
                        ConfusionDict[ConfusionDictKey]++;
                    else
                        ConfusionDict.Add(ConfusionDictKey, 1);
                    Sw1.Write("array:"+totalDoc+"\t");
                    totalDoc++;
                    foreach (var item in orderedDict)
                    {
                        Sw1.Write(item.Key + "\t" + System.Math.Pow(10, item.Value) + "\t");
                    }
                    Sw1.WriteLine();
                }
            }
            int correctPred = 0;
            Console.WriteLine("Confusion matrix for the " + testOrTrain + " data:\n row is the truth, column is the system output");
            Console.Write("\t\t\t");
            foreach (var actClass in UniqueClasses)
            {
                Console.Write(actClass + "\t");
            }
            Console.WriteLine();
            foreach (var actClass in UniqueClasses)
            {
                st1 = actClass;
                Console.Write(st1 + "\t");
                foreach (var predClass in UniqueClasses)
                {
                    st2 = predClass;
                    if (ConfusionDict.ContainsKey(st1 + "_" + st2))
                    {
                        Console.Write(ConfusionDict[st1 + "_" + st2] + "\t");
                        if (st1 == st2)
                            correctPred += ConfusionDict[st1 + "_" + st2];
                    }
                    else
                        Console.Write("0" + "\t");

                }
                Console.WriteLine();
            }
            Console.WriteLine(testOrTrain + " accuracy=" + Convert.ToString(correctPred / totalDoc));
            Console.WriteLine();
            Sw1.Close();
        }
        public static Dictionary<string, List<Document>> ReadData (string TrainingDataPath, ref int Docid, bool training, ref Dictionary<string, bool> Vocab)
        {
            string line, key;
            int value, index;
            Dictionary<string, List<Document>> TrainingClassDict = new Dictionary<string, List<Document>>();

            using (StreamReader Sr = new StreamReader(TrainingDataPath))
            {
                while ((line = Sr.ReadLine()) != null)
                {
                    if (String.IsNullOrWhiteSpace(line))
                        continue;
                    string[] words = line.Split(new string[] { " " }, StringSplitOptions.RemoveEmptyEntries);
                    Document temp = new Document(Docid++);
                    for (int i = 1; i < words.Length; i++)
                    {
                        index = words[i].IndexOf(":");
                        key = words[i].Substring(0, index);
                        value = Convert.ToInt32(words[i].Substring(index + 1));
                        if (temp.WordCount.ContainsKey(key))
                            temp.WordCount[key] += value;
                        else
                            temp.WordCount.Add(key, value);
                        if (training)
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
