using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Bayes
{
    class ProbClass
    {
        public string classLabel;
        public double prob;


        public ProbClass ()
        {
            classLabel = "";
            prob = double.MinValue;
            
        }
        public ProbClass (string ClassLabel, double Prob)
        {
            classLabel = ClassLabel;
            prob = Prob;
        }
    }
}
