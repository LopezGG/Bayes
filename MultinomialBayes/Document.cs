using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MultinomialBayes
{
    class Document
    {
        public int Doc_id;
        public Dictionary<String, int> WordCount;

        public Document ()
        {
            WordCount = new Dictionary<string, int>();
            Doc_id = -1;
        }
        public Document (int Did)
        {
            WordCount = new Dictionary<string, int>();
            Doc_id = Did;
        }
    }
}
