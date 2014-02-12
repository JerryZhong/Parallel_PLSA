#include <fstream>
#include <sstream>
#include <algorithm>
#include "iPLSA.hpp"
#include "Schedule.hpp"
#include "time.h"
#include <float.h>		// for -DBL_MAX
#include <string.h>
#include <omp.h>
#include "AlignAlloc.h"
#include "emmintrin.h"
#include "CBScheduler.hpp"

__m128d _mm_load_pd(double const *dp);
void _mm_store_pd(double *dp, __m128d a);
__m128d _mm_mul_pd(__m128d a, __m128d b);
__m128d _mm_add_pd(__m128d a, __m128d b);
__m128d _mm_setzero_pd(void);

const int align=6;

template<class T>
T * alignedNew(int n){
	return (T*)(AlignAlloc::al_alloc(sizeof(T)*n));
}

template<class T>
void alignedDelete(T* p){
	AlignAlloc::al_free(p);
}

bool iPLSA::loadMatrixFile()
{
	ifstream matFile( matFileName.c_str() );
	// should we handle exception here?
	if ( !matFile.good() )
	{
		cerr << "Error opening matrix file." << endl;
		matFile.close();
		return false;
	}

	matFile >> sizeD;
	matFile >> sizeW;
	// the number of non-zero elements in the matrix
	int numNonZero;
	matFile >> numNonZero;

	// set up data and testData.
	vector<int> * dataFreq = new vector<int>[ sizeW ];
	vector<int> * dataDocID = new vector<int>[ sizeW ];
	vector<int> * testDataFreq = new vector<int>[ sizeW ];
	vector<int> * testDataDocID = new vector<int>[ sizeW ];

	for(int d = 0; d < sizeD; d++) 
	{
		// read a line from the matrix file
		char ch;
		int termID;
		float ffreq;
		bool currentLine = true;
		do
		{
			matFile >> termID;
			termID--; // the termID of the file begins at 1, but we need it to begin with 0

			matFile >> ffreq;
			int freq = ( int ) ffreq;
			--numNonZero;
			int r = rand();
			if ( ( r % testPercentage ) == 0 )
			{
				testDataFreq[ termID ].push_back( freq );
				testDataDocID[ termID ].push_back( d );
			}
			else
			{
				dataFreq[ termID ].push_back( freq );
				dataDocID[ termID ].push_back( d );
			}

			while ( 1 )
			{
				matFile.get( ch );
				if ( ch == '\n' )
				{    // go to the next line
					currentLine = false;
                    			break;
				}
				else if ( ch >= '0' && ch <= '9' )
				{ // ch is a number, go to the next term
					matFile.putback( ch );
					break;
                		}
            		}
		}while ( matFile.good() && currentLine );
	}
	divideTrainData(numBlocks,dataFreq,dataDocID,freqs,docIDs,rowStarts,wStart,dStart);
	delete[] dataFreq;
	delete[] dataDocID;
	divideTestData(numTestBlocks,testDataFreq,testDataDocID,testFreqs,testDocIDs,testRowStarts);
	delete[] testDataFreq;
	delete[] testDataDocID;

	if ( numNonZero != 0 )
	{
		matFile.close();
		//cerr << "Number of non-zero elements in matrix file is wrong!" << endl;
		//cerr << "numNonZero: " << numNonZero << endl;
		return false;
	}
	matFile.close();
	return true;
}

void iPLSA::init() 
{
	int d, w;
	p_z_prev = alignedNew<double>( sizeZ+align );	
	p_z_current = alignedNew<double>( sizeZ+align );
	p_z_best = alignedNew<double>( sizeZ+align );

	p_w_z_prev = alignedNew<double * >( sizeW );
	p_w_z_current = alignedNew<double * >( sizeW );
	p_w_z_best = alignedNew<double * >( sizeW );

	p_d_z_prev = alignedNew<double * >( sizeD );
	p_d_z_current = alignedNew<double * >( sizeD );
	p_d_z_best = alignedNew<double * >( sizeD );

	double *tmp1 = alignedNew<double>( ( sizeZ+align ) * sizeW );
	double *tmp2 = alignedNew<double>( ( sizeZ+align ) * sizeW );
	double *tmp3 = alignedNew<double>( ( sizeZ+align ) * sizeW );
	for ( w = 0; w < sizeW; w++ )
	{
		p_w_z_prev[ w ] = tmp1;
		p_w_z_current[ w ] = tmp2;
		p_w_z_best[ w ] = tmp3;
		tmp1 += sizeZ+align;
		tmp2 += sizeZ+align;
		tmp3 += sizeZ+align;
	}

	tmp1 = alignedNew<double>( ( sizeZ+align ) * sizeD );
	tmp2 = alignedNew<double>( ( sizeZ+align ) * sizeD );
	tmp3 = alignedNew<double>( ( sizeZ+align ) * sizeD );
	for ( d = 0; d < sizeD; d++ )
	{
		p_d_z_prev[ d ] = tmp1;
		p_d_z_current[ d ] = tmp2;
		p_d_z_best[ d ] = tmp3;
		tmp1 += sizeZ+align;
		tmp2 += sizeZ+align;
		tmp3 += sizeZ+align;
	}
}

void iPLSA::divideTestData(const int _numBlocks, const vector<int> * const freqVectors, const vector<int> * const docIDVectors, int **& _freqs,int **& _docIDs, int **& _rowStarts)
{
	_docIDs = alignedNew<int *>( _numBlocks );
	_freqs = alignedNew<int *>( _numBlocks );
	_rowStarts = alignedNew<int *>( _numBlocks );

	int * docIDStart = new int[ _numBlocks + 1 ];
	int docID = 0;
	for ( int i = 0;i < _numBlocks;i++ )
	{
		docIDStart[ i ] = docID;
		docID += sizeD / _numBlocks;
	}
	docIDStart[ _numBlocks ] = sizeD;

	int * numElmts = new int[ _numBlocks ];
	int * index = new int[ _numBlocks ];
	for ( int i = 0;i < _numBlocks;i++ )
	{
		numElmts[ i ] = 0;
		index[ i ] = 0;
		_rowStarts[ i ] = alignedNew<int>( sizeW + 1 );
	}

	for ( int w = 0;w < sizeW;w++ )
	{
		for ( int i = 0;i < ( int ) docIDVectors[ w ].size();i++ )
		{
			int d = docIDVectors[ w ][ i ];
			int blockNum;
			for ( blockNum = 0;blockNum < _numBlocks;blockNum++ )
				if ( docIDStart[ blockNum + 1 ] > d )
					break;
			numElmts[ blockNum ] ++;
		}
	}

	for ( int i = 0;i < _numBlocks;i++ )
	{
		int n = numElmts[ i ];
		_freqs[ i ] = alignedNew<int>( n );
		_docIDs[ i ] = alignedNew<int>( n );
	}

	for ( int w = 0;w < sizeW;w++ )
	{
		for ( int i = 0;i < _numBlocks;i++ )
			_rowStarts[ i ][ w ] = index[ i ];
		for ( int i = 0;i < ( int ) docIDVectors[ w ].size();i++ )
		{
			int d = docIDVectors[ w ][ i ];
			int freq = freqVectors[ w ][ i ];
			int blockNum;
			for ( blockNum = 0;blockNum < _numBlocks;blockNum++ )
				if ( docIDStart[ blockNum + 1 ] > d )
					break;
			int idx = index[ blockNum ];
			_freqs[ blockNum ][ idx ] = freq;
			_docIDs[ blockNum ][ idx ] = d;
			index[ blockNum ] ++;
		}
	}

	for ( int i = 0;i < _numBlocks;i++ )
		_rowStarts[ i ][ sizeW ] = numElmts[ i ];

	delete[] docIDStart;
	delete[] numElmts;
	delete[] index;
}

void iPLSA::divideTrainData(const int _numBlocks, const vector<int> * freqVectors, const vector<int> *docIDVectors, int **& _freqs,int **& _docIDs, int **& _rowStarts, int *& _wStart, int *& _dStart)
{
	int numParts=_numBlocks*_numBlocks;
	_docIDs=alignedNew<int *>(numParts);//ԭ���ǰ��մʵĸ�������ȷ�������ڰ������N*N�и�
	_freqs=alignedNew<int *>(numParts);
	_rowStarts=alignedNew<int *>(numParts);
	_wStart=alignedNew<int>(_numBlocks+1);
	_dStart=alignedNew<int>(_numBlocks+1);

	// count the number of nonzero elements in each row/column
	// containing docoment on each row;    w-d
	// count the number of nonzero elements in each row/column
	int * numElmtsInRow=new int[sizeW];
	for(int i=0;i<sizeW;i++)
		numElmtsInRow[i]=0;
	int * numElmtsInCol=new int[sizeD];
	for(int i=0;i<sizeD;i++)
		numElmtsInCol[i]=0;
	int totalNumElmts=0;
	for(int w=0;w<sizeW;w++){
		int n=(int)docIDVectors[w].size();
		numElmtsInRow[w]=n;
		totalNumElmts+=n;
		for(int i=0;i<n;i++){
			int d=docIDVectors[w][i];
			numElmtsInCol[d]++;
		}
	}

	// make each block have nearly the same amount of nonzero elements 
	int w=0;
	_wStart[0]=0;
	for(int i=0;i<_numBlocks;i++){
		int elmts=0;
		while(w<sizeW && elmts < totalNumElmts/_numBlocks){
			elmts+=numElmtsInRow[w];
			w++;
		}
		_wStart[i+1]=w;
	}
	int d=0;
	_dStart[0]=0;
	for(int i=0;i<_numBlocks;i++){
		int elmts=0;
		while(d<sizeD && elmts<totalNumElmts/_numBlocks){
			elmts+=numElmtsInCol[d];
			d++;
		}
		_dStart[i+1]=d;
	}
	// count the number of elements in each block
	int * numElmts=new int[numParts];
	int * index=new int[numParts];
	for(int i=0;i<numParts;i++){
		numElmts[i]=0;
		index[i]=0;
	}
	for(int w=0;w<sizeW;w++){
		int wBlockNum;
		for(wBlockNum=0;wBlockNum<_numBlocks;wBlockNum++)
			if(_wStart[wBlockNum+1]>w)
				break;
		for(int i=0;i<(int)docIDVectors[w].size();i++){
			int d=docIDVectors[w][i];
			int dBlockNum;
			for(dBlockNum=0;dBlockNum<_numBlocks;dBlockNum++)
				if(_dStart[dBlockNum+1]>d)
					break;
			int partNum=wBlockNum*_numBlocks+dBlockNum;
			numElmts[partNum]++;			
		}
	}

	// store the elements in a block continuously 
	for(int i=0;i<_numBlocks;i++){
		for(int j=0;j<_numBlocks;j++){
			int partNum=i*_numBlocks+j;
			_rowStarts[partNum]=alignedNew<int> (_wStart[i+1]-_wStart[i]+1);
		}
	}

	//Ϊÿһ���ֿ������Ӧ�洢�������Ŀռ䣬�洢��Ӧ��Ĵ�Ƶ���ĵ�ID
	for(int i=0;i<numParts;i++){
		int n=numElmts[i];
		_freqs[i]=alignedNew<int>(n);
		_docIDs[i]=alignedNew<int>(n);
	}

	for(int w=0;w<sizeW;w++){
		int wBlockNum;
		for(wBlockNum=0;wBlockNum<_numBlocks;wBlockNum++)
			if(_wStart[wBlockNum+1]>w)
				break;
		for(int i=wBlockNum*_numBlocks;i<(wBlockNum+1)*_numBlocks;i++){
			int wIndex=w-_wStart[wBlockNum];
			_rowStarts[i][wIndex]=index[i];
		}
		for(int i=0;i<(int)docIDVectors[w].size();i++){
			int d=docIDVectors[w][i];
			int freq=freqVectors[w][i];
			int dBlockNum;
			for(dBlockNum=0;dBlockNum<_numBlocks;dBlockNum++)
				if(_dStart[dBlockNum+1]>d)
					break;
			int partNum=wBlockNum*_numBlocks+dBlockNum;
			int idx=index[partNum];
			_freqs[partNum][idx]=freq;
			_docIDs[partNum][idx]=d;
			index[partNum]++;
		}
	}

	for(int i=0;i<numParts;i++){
		int wBlockNum=i/_numBlocks;
		int wIndex=_wStart[wBlockNum+1]-_wStart[wBlockNum];
		_rowStarts[i][wIndex]=numElmts[i];
	}

	delete[] numElmts;
	delete[] index;
	delete[] numElmtsInRow;
	delete[] numElmtsInCol;
}

// Probabilistic Latent Semantic Analysis
// Probabilistic Latent Semantic Analysis ;
//"test.mat"	 0		 2				1		 1	  1beta��ʼֵ	    0.6	��Сbetaֵ	  0	0.92 �任��	    cpu_core_nums	 4	0
iPLSA::iPLSA(const string & fileName,char * indexMidFileName,char* indexTagFileName, int _testPercentage, int numCats, 
			int numIter, int numRestarts, double betastart, double betastop, double anneal, double betaMod, int _numThreads, int _numBlocks,int _pos) :
	matFileName(fileName),MidFileName(indexMidFileName),TagFileName(indexTagFileName), testPercentage(_testPercentage), numberOfIterations(numIter),
	numberOfRestarts(numRestarts), startBeta(betastart), beta(betastart),
	betaMin(betastop), annealcue(anneal), betaModifier(betaMod), 
	sizeZ(numCats), numThreads(_numThreads), numBlocks(_numBlocks),pos(_pos)
{	
	numTestBlocks=1;
	// actually we should throw an exception here
	if ( !loadMatrixFile() )
	{
		cout << "error loading matrix file" << endl;
		exit( 1 );
	}
	init();
	initializeParameters();
	if(pos == 0)
		this->pos = 1;
	else
		this->pos = 16 / _numThreads;
}

iPLSA::~iPLSA() 
{
	alignedDelete(p_w_z_prev[0]);
	alignedDelete(p_w_z_current[0]);
	alignedDelete(p_w_z_best[0]);

	alignedDelete(p_d_z_prev[0]);
	alignedDelete(p_d_z_current[0]);
	alignedDelete(p_d_z_best[0]);

	alignedDelete(p_z_prev);
	alignedDelete(p_w_z_prev);
	alignedDelete(p_d_z_prev);
	alignedDelete(p_z_current);
	alignedDelete(p_w_z_current);
	alignedDelete(p_d_z_current);
	alignedDelete(p_z_best);
	alignedDelete(p_w_z_best);
	alignedDelete(p_d_z_best);

	for(int i=0;i<numBlocks*numBlocks;i++)
	{
		alignedDelete(freqs[i]);
		alignedDelete(docIDs[i]);
		alignedDelete(rowStarts[i]);	
	}

	for(int i=0;i<numTestBlocks;i++)
	{
		alignedDelete(testFreqs[i]);
		alignedDelete(testDocIDs[i]);
		alignedDelete(testRowStarts[i]);		
	}

	alignedDelete(freqs);
	alignedDelete(docIDs);
	alignedDelete(rowStarts);
	alignedDelete(testFreqs);
	alignedDelete(testDocIDs);
	alignedDelete(testRowStarts);
	alignedDelete(wStart);
	alignedDelete(dStart);
}

//��p_w_z_current copy��p_w_z_best����ȥ...
//��p_d_z_current copy��p_d_z_best����ȥ...
//�����ս���������ʱ�򣬵õ��Ĳ�������Ľ��...
void iPLSA::setBestToCurrent() 
{
	int sizeLine=sizeof(double)*sizeZ;
	memcpy(p_z_best,p_z_current,sizeLine);
	for(int i = 0; i < sizeW; i++)
		memcpy(p_w_z_best[i],p_w_z_prev[i],sizeLine); 
	for(int i = 0; i < sizeD; i++)
		memcpy(p_d_z_best[i],p_d_z_prev[i],sizeLine); 
}
   
// This routine gets the ball rolling
void iPLSA::run() 
{
	iterate();
}

// This is the main routine of the model training
void iPLSA::iterate() 
{
	double time=get_time();
	double pTLL=-HUGE_VAL;
	double lA=0;
	beta = startBeta;

	testLikelihood( pTLL, lA, p_z_prev, p_d_z_prev, p_w_z_prev );

	int sizeLine = sizeof( double ) * (sizeZ+align);
	int inc = 1;
	double one = 1.0;
	double ** tmp = alignedNew<double*>( numThreads ); 	     // temporary space to store vectors
	double ** tmp_denominator = alignedNew<double*>(numThreads); // temporary space to store denominator
	double ** p_d_w_128s = alignedNew<double *>( numThreads );

	
	//���̲߳���֮ǰ���͸�ÿһ���߳̿��ٵ�һ���ռ�
	for ( int i = 0;i < numThreads;++i )
	{
		tmp[ i ] = alignedNew<double>( ( sizeZ+align ) );
		tmp_denominator[ i ] = alignedNew<double>( ( sizeZ+align ) );
		p_d_w_128s[ i ] = alignedNew<double>( 2 );
	}

	TwoForwardScheduler scheduler(numBlocks,numBlocks,numThreads);
	#pragma omp parallel
	{
		double total=0;

		int threadNum=omp_get_thread_num();
		Schedule::setAffinity(threadNum*pos);	// set the affinity so that every processor would only visit its private copy of p_d_z

		if(beta!=1)
		{
			#pragma omp for schedule(static,sizeD/numThreads)
			for ( int d = 0;d < sizeD;d++ )
				vdPowx( sizeZ, p_d_z_prev[ d ], beta, p_d_z_prev[ d ] );
			#pragma omp for schedule(static,sizeW/numThreads)
			for ( int w = 0;w < sizeW;w++ )
				vdPowx( sizeZ, p_w_z_prev[ w ], beta, p_w_z_prev[ w ] );
		}

		int iterNum = 0;
		double localBeta = startBeta; //temper-em��ʼbetaֵ
		double prevTestLL = pTLL;
		double currentTestLL;
		double lastA = lA;  	// These two variables help check for
		double newA;		// numerical errors due to overfitting

		//betaMin��betastop��Сֵֹͣ
		while(iterNum < numberOfIterations && localBeta > betaMin) 
		{
			#pragma omp single
			{
				scheduler.initScheduler();
			}
			/////////////////////////////////////////////////////////////////
			// EM
			/////////////////////////////////////////////////////////////////
			memset( tmp[ threadNum ], 0, sizeLine );
			memset( tmp_denominator[ threadNum ], 0, sizeLine );
			int wBlockNum=0;
			int dBlockNum=0;
			while(1){
				CBScheduler::JobResult r;
				double t1=get_time();
				while(1){
					r=scheduler.getJob(wBlockNum,dBlockNum,wBlockNum,dBlockNum);
					if(r==CBScheduler::ok || r==CBScheduler::done)
						break;
				}
				total+=get_time()-t1;
				if(r==CBScheduler::done)
					break;

				int partNum=wBlockNum*numBlocks+dBlockNum;
				int * _rowStarts=rowStarts[partNum];
				int * _docIDs=docIDs[partNum];
				int * _freqs=freqs[partNum];
				for(int wID=wStart[wBlockNum];wID<wStart[wBlockNum+1];wID++){
					int wIndex=wID-wStart[wBlockNum];
					double * denominator=tmp_denominator[threadNum];//�洢����ķ�ĸ,����ռ�Ϊ|K|
					double * p_d_w_128=p_d_w_128s[threadNum];
					double * tmp1=tmp[threadNum];			//�洢����ķ��ӣ�����ռ�Ϊ|K|,��ÿ�θ��µ�p(z)�洢���������...
					// update the current p_d_z and p_w_z
					for(int i=_rowStarts[wIndex];i<_rowStarts[wIndex+1];i++){
						int d=_docIDs[i];
						int freq=_freqs[i];
						p_d_w_128[0]=0;
						p_d_w_128[1]=0;
						__m128d p_z_d_w;
						__m128d p_d_w=_mm_load_pd(p_d_w_128);//�������ֽڵĴ洢�� p(z|d,w)�ķ�ĸ�����Ƽ���p(z|d,w)�ķ�ĸ��������й�һ���õ�...
						__m128d a;
						for(int z=0;z<sizeZ;z+=2){
							p_z_d_w=_mm_load_pd(&(p_w_z_prev[wID][z]));

							a=_mm_load_pd(&(p_d_z_prev[d][z]));
							p_z_d_w=_mm_mul_pd(p_z_d_w,a); //p(d|z)*p(w|z)

							a=_mm_load_pd(&(p_z_prev[z])); //p(z)
							p_z_d_w=_mm_mul_pd(p_z_d_w,a); //�ڸ���(d,w)������£���Ӧ�Ų�ͬ��z ��p(z)p(d|z)*p(w|z)

							p_d_w=_mm_add_pd(p_z_d_w,p_d_w);//�Ѹ���p(z=0)p(d|z=0)*p(w|z=0) + p(z=1)p(d|z=1)*p(w|z=1)+p(z=2)p(d|z=2)*p(w|z=2)������
											//������Ҫ���p(z|d,w)�ķ�ĸ,��ĸ=(p_d_w_0,p_d_w_1)
							_mm_store_pd(&(tmp1[z]),p_z_d_w);//����p(z|d,w)�ķ���,tmp1�洢���ٸ���(d,w)�����,
											 //z�������µ���ʱ�洢p(z)p(d|z)*p(w|z)...
						}
						_mm_store_pd(p_d_w_128,p_d_w);
						p_d_w_128[0]+=p_d_w_128[1];		 //�����ϲ�֮��p(z|d,w)�ķ�ĸ
						p_d_w_128[0]=freq/p_d_w_128[0];		 //n(d,w)*(1/��ĸ)
						p_d_w_128[1]=p_d_w_128[0];		 //SSE ���ݿ��������м���
						p_d_w=_mm_load_pd(p_d_w_128);		 //transfer-��ĸ

						//�����ڵ�ǰ�߳��µ�tmp1����ʼ����ʱ��Ϊ0,�洢p_z_d_wֵ��
						for(int z=0;z<sizeZ;z+=2)
						{
							p_z_d_w=_mm_load_pd(&(tmp1[z]));  //tmp1����Ϊָ��ָ����ʱ�洢�ռ䣬�洢��������������p(z|d,w)����ֵ
							p_z_d_w=_mm_mul_pd(p_z_d_w,p_d_w);//�ڵ�ǰ(d,w)����µĻ�õ�p(z|d,w) * n(d,w)��ֵ .
											  //˵����E���Ľ������������Ҫ�Ľ����������Ҫ�Ľ����M����õĲ������������ֱ�ӽ�
											  //n(d,w)*p(z|d,w)����������...

							a=_mm_load_pd(&(denominator[z]));//����������tmp1������һ������Ϊָ��ָ����ʱ�洢�ռ䣬��ÿ��EM����ʱ
											 //����ʼ��Ϊ0.�����ǵ�һ�γ��֣�
							a=_mm_add_pd(p_z_d_w,a);
							_mm_store_pd(&(denominator[z]),a);//denominator�㼯�˵�ǰthread���������(d,w)Ԫ�ص�[p(z|d,w) * n(d,w)]���ڵ�ǰz�µ��ܺ�.


							a=_mm_load_pd(&(p_w_z_current[wID][z]));//�������ʽ���������������
							a=_mm_add_pd(p_z_d_w,a);
							_mm_store_pd(&(p_w_z_current[wID][z]),a);//p_w_z_current�㼯�˵�ǰthread�������word=wID������(di,wID)Ԫ�ص�[p(z|di,wID) * n(di,wID)]���ڵ�ǰz�µ��ܺ�.


							a=_mm_load_pd(&(p_d_z_current[d][z]));//�������ʽ���������������
							a=_mm_add_pd(p_z_d_w,a);
							_mm_store_pd(&(p_d_z_current[d][z]),a);	//p_d_z_current�㼯�˵�ǰthread�������doc=d������(d,wj)Ԫ�ص�[p(z|d,wj) * n(d,wj)]���ڵ�ǰz�µ��ܺ�
						}
					}//for end һ��(d,w)pair...
				}
				scheduler.finishJob(wBlockNum,dBlockNum);
			}

			#pragma omp barrier
			{}
			#pragma omp single
			{
				//�����̶߳�Ӧ��denominator[z]���,����һ������������(d,w)Ԫ�ص�[p(z|d,w) * n(d,w)]���ڵ�ǰz=k�µ�ֵ���ܺ�
				for(int i=1;i<numThreads;i++)
					daxpy(&sizeZ,&one,tmp_denominator[i],&inc,tmp_denominator[0],&inc);

				// normalize parameters
				// normalize parameters,�Բ���p(z)���й�һ������...
				double sum_pz=dasum(&sizeZ,tmp_denominator[0],&inc);
				double a=1/sum_pz;
				memset(p_z_current,0,sizeLine);
				daxpy(&sizeZ,&a,tmp_denominator[0],&inc,p_z_current,&inc);	// normalize p_z, note that R is not used here
			}

			#pragma omp barrier
			{} 

			#pragma omp for schedule(static,sizeD/numThreads)
			for(int d=0;d<sizeD;d++)
			{
				vdDiv(sizeZ,p_d_z_current[d],tmp_denominator[0],p_d_z_current[d]);
			}

			#pragma omp for schedule(static,sizeW/numThreads)
			for(int w=0;w<sizeW;w++)
			{	// normalize p_w_z
				vdDiv(sizeZ,p_w_z_current[w],tmp_denominator[0],p_w_z_current[w]);
			}
			//////////////////////////////////////////////
			// end of EM
			//////////////////////////////////////////////

			////////////////////////////////////////////////////
			// begin of testLikehood
			////////////////////////////////////////////////////
			tmp[ threadNum ][ 0 ] = 0;							// LL
			tmp_denominator[ threadNum ][ 0 ] = 0;				// aL

			for ( int blockNum = 0;blockNum < numTestBlocks;blockNum++ )
			{
				int * _testRowStarts = testRowStarts[ blockNum ];
				int * _testDocIDs = testDocIDs[ blockNum ];
				int * _testFreqs = testFreqs[ blockNum ];
				//���̶߳�̬�����������100...
				#pragma omp for schedule(dynamic,100)
				for ( int w = 0;w < sizeW;w++ )
				{
					double sum = 0;
					for ( int i = _testRowStarts[ w ];i < _testRowStarts[ w + 1 ];i++ )
					{
						int d = _testDocIDs[ i ];
						int freq = _testFreqs[ i ];
						sum = 0;
						for ( int z = 0;z < sizeZ;z++ )
							sum += p_z_current[ z ] * p_w_z_current[ w ][ z ] * p_d_z_current[ d ][ z ];
						tmp_denominator[ threadNum ][ 0 ] += sum;
						sum = log( sum );
						if ( sum > -HUGE_VAL )
							tmp[ threadNum ][ 0 ] += freq * sum;
					}
				}
			}

			/*������̹߳������ʱ�洢�ռ�tmp[thread][0]���汣����likelihood��ֵ
			��һ���߳���ִ�е��ⲿ�֣���ͳ��..���������Ҫ��һ��...
			���������ڣ�����̲߳���ִ�У��Ƿ�����쳣�������...
			*/
			double LL = 0;
			double aL = 0;
			for ( int i = 0;i < numThreads;i++ )
			{
				LL += tmp[ i ][ 0 ];
				aL += tmp_denominator[ i ][ 0 ];
			}

			currentTestLL = LL;
			int totalEvents = 0;
			for ( int i = 0;i < numTestBlocks;i++ )
				totalEvents += testRowStarts[ i ][ sizeW ];
			newA = aL / totalEvents;

			#pragma omp single
			cerr<<iterNum<<"\t"<<currentTestLL<<"\t"<<prevTestLL<<"\t"<<(currentTestLL-prevTestLL)<<"\t"<<newA<<"\t"<<lastA<<"\t"<<(newA - lastA)<<"\t"<<localBeta<<endl;
			//////////////////////////////////////////////
			// end of likelihood
			//////////////////////////////////////////////
//			if ( ( currentTestLL - annealcue ) < prevTestLL || newA < lastA )
			if ( ( currentTestLL - annealcue ) < prevTestLL )
			{
				localBeta *= betaModifier;//һ�ε���֮��ı仯ѧϰ��
				currentTestLL = prevTestLL;
				iterNum--;		  //���µ���������betaֵ��betastopС�Ż��˳�whileѭ��...

				#pragma omp for schedule(static,sizeD/numThreads)
				for ( int d = 0;d < sizeD;d++ )
					vdPowx( sizeZ, p_d_z_prev[ d ], betaModifier, p_d_z_prev[ d ] );
				#pragma omp for schedule(static,sizeW/numThreads)
				for ( int w = 0;w < sizeW;w++ )
					vdPowx( sizeZ, p_w_z_prev[ w ], betaModifier, p_w_z_prev[ w ] );
			}//ÿһ�ε�����Ȼ�������ڲ�ͣ������...
			else
			{
				lastA = newA;
				#pragma omp single
				{
					////////////////////////////////////////////////////
					// begin of setPrevToCurrent
					////////////////////////////////////////////////////
					double * tmp;
					tmp = p_z_prev;
					p_z_prev = p_z_current;
					p_z_current = tmp;

					double ** tmp1;
					tmp1 = p_w_z_prev;
					p_w_z_prev = p_w_z_current;
					p_w_z_current= tmp1;
					memset(p_w_z_current[0], 0, ( sizeZ+align ) * sizeW * sizeof(double) );

					tmp1 = p_d_z_prev;
					p_d_z_prev = p_d_z_current;
					p_d_z_current = tmp1;
					memset(p_d_z_current[0], 0, ( sizeZ+align ) * sizeD * sizeof(double) );
				}

				#pragma omp barrier
				{}

				////////////////////////////////////////////////////
				// end of setPrevToCurrent
				////////////////////////////////////////////////////

				#pragma omp for schedule(static,sizeD/numThreads)
				for ( int d = 0;d < sizeD;d++ )
					vdPowx( sizeZ, p_d_z_prev[ d ], localBeta, p_d_z_prev[ d ] );

				#pragma omp for schedule(static,sizeW/numThreads)
				for ( int w = 0;w < sizeW;w++ )
					vdPowx( sizeZ, p_w_z_prev[ w ], localBeta, p_w_z_prev[ w ] );
				prevTestLL = currentTestLL;
			}
			iterNum++;
		}//��ʾ�����Ĵ����Լ��˳�����while(iterNum < numberOfIterations && localBeta > betaMin) 

		if ( newA > lastA && currentTestLL > prevTestLL )
		{
			#pragma omp single
			setBestToCurrent();
		}
		else
		{
			// because the prev matrices have been performed power operation, we should restore them
			#pragma omp for schedule(static,sizeD/numThreads)
			for ( int d = 0;d < sizeD;d++ )
				vdPowx( sizeZ, p_d_z_prev[ d ], 1 / localBeta, p_d_z_best[ d ] );

			#pragma	omp for schedule(static,sizeW/numThreads)
			for ( int w = 0;w < sizeW;w++ )
				vdPowx( sizeZ, p_w_z_prev[ w ], 1 / localBeta, p_w_z_best[ w ] );

			#pragma omp single
			memcpy( p_z_best, p_z_prev, sizeof( double ) * sizeZ );

			#pragma omp barrier
        		{}
         	}

		 //normalize p_d_z distribution
		 #pragma omp for schedule(static,sizeD/numThreads)
		for ( int d = 0;d < sizeD;d++ )
		{
			double tmpsum=0.0;
			for(int z=0;z<sizeZ;z++)
				tmpsum += p_d_z_best[d][z]; 
			for(int z=0;z<sizeZ;z++) 
				p_d_z_best[d][z] /=tmpsum; 
		}

	} // end of parallel region

	for ( int i = 0;i < numThreads;++i )
	{
		alignedDelete( tmp[ i ] );
		alignedDelete( tmp_denominator[ i ] );
		alignedDelete( p_d_w_128s[ i ] );
	}
	alignedDelete( tmp );
	alignedDelete( tmp_denominator );
	alignedDelete( p_d_w_128s );

	cerr<<"time used:"<<get_time()-time<<endl;
}

void iPLSA::initializeParameters() 
{
	// initializing prev
	// Make sure we can use drand48 w/ win...
	// initialize p_z_prev
	srand48((unsigned)time(0));
	int inc = 1;
	double sum = 0.0;
	for ( int i = 0; i < sizeZ; i++ )
	{
		double r = drand48() * drand48();
		p_z_prev[ i ] = r;
		sum += r;
	}
	sum = 1 / sum;
	dscal( &sizeZ, &sum, p_z_prev, &inc );
	// initialize p_w_z_prev
	for ( int j = 0; j < sizeZ; j++ )
	{
		sum = 0.0;
		for ( int i = 0; i < sizeW; i++ )
		{
			double rnd = drand48() * drand48();
			sum += rnd;
			p_w_z_prev[ i ][ j ] = rnd;
		}
		for ( int i = 0; i < sizeW; i++ )
			p_w_z_prev[ i ][ j ] = p_w_z_prev[ i ][ j ] / sum;
	}
	// initialize p_d_z_prev
	for ( int j = 0; j < sizeZ; j++ )
	{
		sum = 0.0;
		for ( int i = 0; i < sizeD; i++ )
		{
			double rnd = drand48() * drand48();
			sum += rnd;
			p_d_z_prev[ i ][ j ] = rnd;
		}
		for ( int i = 0; i < sizeD; i++ )
			p_d_z_prev[ i ][ j ] = p_d_z_prev[ i ][ j ] / sum;
	}
}

void iPLSA::testLikelihood( double &logLikelihood, double &averageLikelihood, double * p_z, double ** p_d_z, double ** p_w_z )
{
    // Calculate the held out data log-liklihood using current parameters
    double LL = 0.0;
    // Calculate the average likelihood of an event
    double aL = 0;
#pragma omp parallel
    {
        int threadNum = omp_get_thread_num();
        Schedule::setAffinity( threadNum * pos);

        for ( int blockNum = 0;blockNum < numTestBlocks;blockNum++ )
        {
            int * _testRowStarts = testRowStarts[ blockNum ];
            int * _testDocIDs = testDocIDs[ blockNum ];
            int * _testFreqs = testFreqs[ blockNum ];
	//���߳�ִ�е�ʱ��ÿһ���̸߳�LL,aL������ʼֵ,������3���̼߳�  LL0=LL1=LL2=AL0=AL1=AL2=0
	//Ȼ���м������֮��LL=LL0+LL1+LL2,AL=AL0+AL1+AL2;����ִ�м���...
	//LL:����������Ҫ�������log-likelihood.   LL=n(d,w)logP(d,w)��ʽ
	//AL:��ֻ��   p(d,w)���...
		#pragma omp for reduction(+:LL,aL)
	//��ÿһ��(word-doc)���󣬽����и����ִ��������
            for ( int w = 0;w < sizeW;w++ )
            {
                double sum = 0;
                double pLL = 0;
                double paL = 0;
                for ( int i = _testRowStarts[ w ];i < _testRowStarts[ w + 1 ];i++ )
                {
                    int d = _testDocIDs[ i ];
                    int freq = _testFreqs[ i ];
                    sum = 0;
                    for ( int z = 0;z < sizeZ;z++ )
                        sum += p_z[ z ] * p_w_z[ w ][ z ] * p_d_z[ d ][ z ];
                    paL += sum;
                    sum = log( sum );
                    if ( sum > -HUGE_VAL )
                        pLL += freq * sum;
                }
                LL += pLL;
                aL += paL;
            }
        }
    }
    int totalEvents = 0;
    for ( int i = 0;i < numTestBlocks;i++ )
        totalEvents += testRowStarts[ i ][ sizeW ];
    logLikelihood = LL;
    //�ܹ��е�Ԫ��(d,w)������(P(d,w)���)/�ܸ���
    averageLikelihood = aL / totalEvents;
}
