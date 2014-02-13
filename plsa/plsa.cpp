#include <iostream>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

using namespace std;

struct feature_node
{
        int index;
        int value;
	double* iterate_topic;
};

struct problem
{
	int l;
	long *mid;
	struct feature_node **x;
	long elements;
};

const long document_total_number = 2;//80 * 1000;
const long topic_total_number = 4;//200;
const long vocabulary_total_number = 10;//800 * 1000;
const long segment_len = 50;

struct problem prob;
feature_node * x_space = NULL;

double* doc_topic = NULL;  //document to topic distribution
double* topic_word = NULL; //topic to word distribution
char* vocabulary = NULL;

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
        int len;

        if(fgets(line,max_line_len,input) == NULL)
                return NULL;

        while(strrchr(line,'\n') == NULL)
        {
                max_line_len *= 2;
                line = (char *) realloc(line,max_line_len);
                len = (int) strlen(line);
                if(fgets(line+len,max_line_len-len,input) == NULL)
                        break;
        }
        return line;
}

int randomProbilities(double probability[],int size)
{
        srand(time(NULL));

        double total = 0;
        for (int i = 0; i < size; i++)
        {
            probability[i] = rand() + 1;
            total += probability[i];
        }

        //normalize
        for (int i = 0; i < size; i++)
        {
            probability[i] = probability[i] / total;
        }

        return 0;
}

int read_co_ocurrence(FILE* fp)
{
	int elements = 0, max_index = 0;
	char *endptr = NULL;
	char *idx = NULL, *val = NULL;

	prob.l = 0;
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t");

		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n')
				break;
			elements++;
		}

		prob.l ++;
	}
	rewind(fp);

	prob.mid = Malloc(long,prob.l);
	prob.x = Malloc(struct feature_node *,prob.l);
	x_space = Malloc(struct feature_node,elements+prob.l);
	prob.elements = elements+prob.l;

	max_index = 0;
	int j=0;
	for(int i=0;i<prob.l;i++)
	{
		readline(fp);

		prob.x[i] = &x_space[j];
		char* mid = strtok(line,"\t");
		prob.mid[i] = strtol(mid,&endptr,10);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			x_space[j].index = (int) strtol(idx,&endptr,10);
			x_space[j].value = strtol(val,&endptr,10);
			x_space[j].iterate_topic = new double[topic_total_number];

			++j;
		}

		x_space[j++].index = -1;
	}
	return 0;
}

int read_vocabulary(FILE* fp)
{
	char *idx,*val,*endptr;

        while(readline(fp)!=NULL)
        {
              idx = strtok(line,"\t");
		val = strtok(NULL,"\n");

		if( val == NULL )
			break;

		long index = (long)strtol(idx,&endptr,10 );
		strcpy( &vocabulary[index * segment_len] , val );
        }

	return 0;
}

feature_node* find_word_in_document(int docindex,int wordindex)
{
	int doc_feature_node_num = 0;
	feature_node* cur = NULL;

	//it obtains document word's number by the formulation  such that document-word total memory dividing unit of feature node
	feature_node* next_feature_node = NULL;
	if(docindex >= prob.l - 1)
		next_feature_node = prob.x[0] + prob.elements;
	else
		next_feature_node =  prob.x[docindex + 1];

	doc_feature_node_num = next_feature_node - prob.x[docindex] - 1;

	
	int left = 0;
    	int right = doc_feature_node_num;
    	int half = (left+right) / 2;
    	while(left <= right)
    	{
		if(prob.x[docindex][half].index == wordindex)
			return prob.x[docindex] + half;

		if(prob.x[docindex][half].index < wordindex)
			left = half + 1;
		else
			right = half - 1;

        	half = (left + right) / 2;
    	}

	return NULL;
}

int EM()
{
	double total = 0.0 ;

	//E STEP
	feature_node* docnode = NULL;
	for(int docIndex = 0 ; docIndex < prob.l ; docIndex++ )
	{
		for(docnode = prob.x[docIndex]; docnode->index != -1;docnode++)
		{
			int wordIndex = docnode->index;
			total = 0.0 ;
			double* iterate_topic = docnode->iterate_topic;

			for(int topicIndex=0; topicIndex < topic_total_number ; topicIndex++)
			{
				double numerator = doc_topic[docIndex * topic_total_number + topicIndex] * topic_word[topicIndex * vocabulary_total_number + wordIndex];
				total += numerator;
				iterate_topic[topicIndex] = numerator;
			}

			for(int topicIndex = 0; topicIndex < topic_total_number; topicIndex++)
			{
				iterate_topic[topicIndex] = iterate_topic[topicIndex] / total;
			}
		}
	}

	//M step
	//P(W|Z)
	for(int topicIndex = 0 ;topicIndex < topic_total_number ;topicIndex++)
	{
		double totalDenominator = 0.0 ; 
		for(int wordIndex =0 ; wordIndex < vocabulary_total_number; wordIndex++)
		{
			double numerator = 0.0;
			for(int docIndex = 0 ; docIndex < prob.l ; docIndex++ )
			{	
				feature_node *docnode = find_word_in_document(docIndex,wordIndex);
				if(NULL == docnode)
					continue;

				numerator += docnode->value * docnode->iterate_topic[topicIndex];
			}
			topic_word[topicIndex * vocabulary_total_number + wordIndex] = numerator;
			totalDenominator += numerator ;
		}
		for(int wordIndex=0; wordIndex < vocabulary_total_number; wordIndex++)
			topic_word[topicIndex*vocabulary_total_number + wordIndex] = topic_word[topicIndex*vocabulary_total_number + wordIndex] / totalDenominator;
	}

	//P(Z|D)
	for(int docIndex=0;docIndex<document_total_number;docIndex++)
	{
		double totalDenominator = 0.0 ;
		for( int topicIndex =0 ;topicIndex < topic_total_number; topicIndex++)
		{
			double numerator = 0.0;
			for ( int wordIndex = 0 ; wordIndex < vocabulary_total_number ; wordIndex++ )
			{
				feature_node *docnode = find_word_in_document(docIndex,wordIndex);
				if(NULL == docnode)
					continue;

				numerator += docnode->value * docnode->iterate_topic[topicIndex];
			}
			doc_topic[docIndex * topic_total_number + topicIndex] = numerator;
			totalDenominator += numerator;
		}

		for(int topicIndex=0;topicIndex<topic_total_number;topicIndex++)
			doc_topic[docIndex * topic_total_number + topicIndex] = doc_topic[docIndex * topic_total_number + topicIndex] / totalDenominator;
	}

	return 1;
}

int main(int argc,char* argv[])
{
        int index = 0;

        doc_topic = new double[document_total_number * topic_total_number];
        randomProbilities(doc_topic,document_total_number * topic_total_number);

        topic_word = new double[ topic_total_number * vocabulary_total_number];
        randomProbilities(topic_word, topic_total_number * vocabulary_total_number);

        vocabulary = new char[vocabulary_total_number * segment_len];
        for(index = 0 ; index < vocabulary_total_number * segment_len ; ++index)
                vocabulary[index] = 0;

	max_line_len = 1024;
        line = Malloc(char,max_line_len);

       	FILE *fp = fopen("./data/indextag.txt","r");
	read_vocabulary(fp);

	fp = fopen("./data/svmformattags.txt","r");
	read_co_ocurrence(fp);

	EM();

        return 0;
}
