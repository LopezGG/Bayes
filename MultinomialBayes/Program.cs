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
            double classPriorDelta = Convert.ToDouble(args[2]);
            double condProbDelta = Convert.ToDouble(args[3]);
            string modelFile = args[4];
            string sysOutput = args[5];
            int Docid = 0;
            Dictionary<string, bool> Vocab = new Dictionary<string, bool>();
            //read docs
            Dictionary<string, List<Document>> TrainingDocs = ReadData(TrainingDataPath, ref Docid, true, ref Vocab);
            double totalTrainingDocs = Docid;
            double DocsPerClass = 0;
            string featureClass;
            double prob = 0;
            Dictionary<string, double> classPrior = new Dictionary<string, double>();
            Dictionary<string, double> FeatureClassProb = new Dictionary<string, double>();
            int totalClass = TrainingDocs.Count;
            //calculating Priors
            foreach (var classLabel in TrainingDocs)
            {
                DocsPerClass = classLabel.Value.Count;
                //calculate Prior
                prob = System.Math.Log10((DocsPerClass + classPriorDelta) / (totalTrainingDocs + (classPriorDelta * totalClass)));
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
                    FeatureClassProb.Add(featureClass, logProb);

                }

            }

            ClassifyandWrite(sysOutput, TrainingDocs, Vocab, FeatureClassProb, classPrior, "train");
            Dictionary<string, List<Document>> TestingDocs = ReadData(TestingDataPath, ref Docid, false, ref Vocab);
            ClassifyandWrite(sysOutput, TestingDocs, Vocab, FeatureClassProb, classPrior, "test");
            Console.WriteLine("Done");
            Console.ReadLine();
        }
        public static void ClassifyandWrite (string sysOutput, Dictionary<string, List<Document>> TrainingDocs, Dictionary<string, bool> Vocab, Dictionary<string, double> FeatureClassProb, Dictionary<string, double> classPrior, string testOrTrain)
        {
            StreamWriter Sw1 = new StreamWriter(sysOutput);
            string featureClass;
            Dictionary<String, int> ConfusionDict = new Dictionary<string, int>();
            string st1, st2;
            var UniqueClasses = TrainingDocs.Keys.ToList();
            Dictionary<int, List<ProbClass>> DocClassProb = new Dictionary<int, List<ProbClass>>();
            string ConfusionDictKey;
            double totalDoc = 0;
            foreach (var classGroup in TrainingDocs)
            {

                foreach (var document in classGroup.Value)
                {
                    totalDoc++;
                    ProbClass maxProbClass = new ProbClass();
                    foreach (var genClass in UniqueClasses)
                    {
                        double Docprob = 0;

                        foreach (var word in Vocab)
                        {
                            featureClass = genClass + "_" + word.Key;
                            if (document.WordCount.ContainsKey(word.Key))
                                Docprob += (document.WordCount[word.Key] * FeatureClassProb[featureClass]);
                        }
                        Docprob += classPrior[genClass];
                        if (Docprob > maxProbClass.prob)
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
