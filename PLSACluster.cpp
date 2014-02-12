#include <iostream>
#include <stdlib.h>
#include <string> 
#include <fstream>
#include <sstream>
#include "iPLSA.hpp"
#include <omp.h>
#include <map>  
#include <vector>
#include <stdio.h>
#include <algorithm>

int cmp(const pair<int, double>& x, const pair<int, double>& y)
{
	return  x.second > y.second;	
}

void sortMapByValue(map<int, double>& tMap, vector<pair<int, double> >& tVector)
{
	for (map<int, double>::iterator curr = tMap.begin(); curr != tMap.end(); curr++)
	{ 
		tVector.push_back(make_pair(curr->first, curr->second)); 
	}	
	sort(tVector.begin(), tVector.end(), cmp);	
}


int loadmidinfo(ifstream &in , vector<int> & index2mid)
{
	int index;
	int mid;
	string s;
    while(getline(in,s))
	{
		istringstream in_str(s);
		in_str>>index>>mid;
		index2mid[index]=mid;
    }
	return 0;
}


int loadtaginfo(ifstream &in , vector<string> & index2tag)
{
	int index;
	string tag;
	string s;
    while(getline(in,s))
	{
		istringstream in_str(s);
		in_str>>index>>tag;
		index2tag[index]=(tag); 
    }
	return 0;
}


int main(int argc, char * argv[])
{
	if(argc!= 11)
	{
		cout<<"usage: PLSACluster <inputfile>  <indexmidfile> <indextagfile> <crossfolds> <numTopics> <numIters> <anneal> <numBlocks> <top-k words> <pos>"<<endl;
		cout<<"./PLSACluster  data/inputtagsformat.txt  data/indexmediaid.txt  data/indextag.txt 10 200 200 100 8 50 0"<<endl;
		return 1;
	}

	char* inputfile=argv[1];		// input file
	char* indexmidfile=argv[2];		// mid inverted index table file
	char* indextagfile=argv[3];		// tag inverted index table file
	int crossfold=atoi(argv[4]);	// cross validation dataset  10(1:9)
	int numLS=atoi(argv[5]);		// topic number
	int numIters=atoi(argv[6]);		// iterate number
	int anneal=atoi(argv[7]);		// simulated annealing
	int numBlocks=atoi(argv[8]);	// block number
	int topk=atoi(argv[9]);			// number of tags in each topics
	int pos=atoi(argv[10]);

	int cpu_core_nums = omp_get_num_procs();
	omp_set_num_threads(cpu_core_nums);
	
	iPLSA * plsa;

	plsa=new iPLSA(inputfile,indexmidfile,indextagfile,crossfold, numLS, numIters, 1, 1, 0.552, anneal, 0.92, cpu_core_nums, numBlocks, pos);

	plsa->run();

	double ** p_d_z = plsa->get_p_d_z();
	double ** p_w_z = plsa->get_p_w_z();
	int document_num = plsa->numDocs();
	int topic_num = plsa->numCats();
	int word_num = plsa->numWords(); 
	int midcount = plsa->numDocs();

	vector<int>     index2mid(midcount);
	vector<string>  index2tag(word_num);
	ifstream in_inter(indexmidfile);
	ifstream in_inter2(indextagfile);
	loadmidinfo(in_inter,index2mid);
	loadtaginfo(in_inter2,index2tag);

	FILE *doc2topic_fp = fopen("doc2topic_distribution.txt","w");
	if(doc2topic_fp==NULL) return -1;


	for( int i = 0; i < document_num; ++i )
	{
		fprintf(doc2topic_fp, "%d ", index2mid[i]);
		for( int j = 1; j < topic_num; ++j )
		{
			fprintf(doc2topic_fp, "%f ", p_d_z[i][j]);
		}
		fprintf(doc2topic_fp, "\n");
	}

	FILE *topic2word_fp = fopen("topic2word_distribution.txt","w");
	if(doc2topic_fp==NULL) 
		return -1;
	for( int i = 0; i < topic_num; ++i )
	{
		map<int,double> wMap;
		for( int w = 0; w<word_num; w++ )
		{
			wMap[w] = p_w_z[w][i];
		}

		vector< pair<int, double> > wVector;
		sortMapByValue(wMap,wVector);
		for( int w = 1; w<=topk; w++ )
		{
			fprintf(topic2word_fp, "%s:%f ",index2tag[wVector[w].first].c_str(), wVector[w].second); 
		}
		fprintf(topic2word_fp, "\n");
	}

	return 0;
}
