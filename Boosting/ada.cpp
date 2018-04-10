#include <bits/stdc++.h>
#include <time.h>
using namespace std;
#define ff first
#define ss second
#define mp make_pair
#define pb push_back
#define L 200 // number of classifiers
#define T 32561 // number of training examples
#define Tst 16281 //number of testing examples
#define num_samples 750 //number of instances for each classifier
#define no_attr 14

typedef struct node
{
    vector<node*> child;
    vector<int> left;
    int used[20];
    int attr_no;
    int out;
}node;
double coeffs[L],weights[T];
node classifiers[L];
int training_examples[T][no_attr], training_output[T], samples[T];
int testing_examples[Tst][no_attr], testing_output[Tst];
vector<int> attr_type[no_attr];
int cnt_edges;
int samples_training[T];
int samples_current[num_samples][no_attr], output_current[num_samples];
void init()
{
	attr_type[0].resize(4); // continous
	attr_type[1].resize(8);
	attr_type[2].resize(2); // continous
	attr_type[3].resize(16);
	attr_type[4].resize(2); // continous
	attr_type[5].resize(7);
	attr_type[6].resize(14);
	attr_type[7].resize(6);
	attr_type[8].resize(5);
	attr_type[9].resize(2);
	attr_type[10].resize(2); // continous
	attr_type[11].resize(2); // continous
	attr_type[12].resize(2); // continous
	attr_type[13].resize(41);
	for(int i=0;i<no_attr;i++)
	        for(int j=0;j<attr_type[i].size();j++)
        		attr_type[i][j]=j;
}

void init_root(node *root)
{
    	(root->left).resize(num_samples);
	root->out=-1;
	for(int i=0;i<14;i++) root->used[i]=0;
    	for(int i=0;i<num_samples;i++) root->left[i]=i;
}

double information_gain(int sample_current[][no_attr], int output_current[no_attr], vector <int> left , int attr){
    	double info_gain = 0;
    	int num_pos = 0 , num_neg = 0;
    	for(int i = 0 ; i < left.size() ; i++)
	{
        	int inp_ind = left[i];
        	if(output_current[inp_ind]==1)num_pos++;
        	else num_neg++;
    	}
    	double p_pos = (num_pos*1.0)/(left.size()*1.0);
    	double p_neg = (num_neg*1.0)/(left.size()*1.0);
    	double initial_entropy = -1.0 * ( (p_pos*log(p_pos)) + (p_neg*log(p_neg)) );
    	if(num_pos == 0 || num_neg == 0)
	{
        	initial_entropy = 0; //  pure subset , hence leaf node
    	}
    	// now calculate the weighted average entropy after split
    	double weighted_av = 0;
    	for(int i = 0 ; i < attr_type[attr].size() ; i++ )
	{
        	num_pos = 0;
        	num_neg = 0;
        	for(int j = 0 ; j < left.size() ; j++)
		{
        		int inp_ind = left[j];
            		if(sample_current[inp_ind][attr]==attr_type[attr][i])
			{
	                	if(output_current[inp_ind]==1)num_pos++;
	                	else num_neg++;
	            	}
	        }
	        int tot = num_pos + num_neg ;
	        p_pos = (num_pos*1.0)/(tot*1.0);
	        p_neg = (num_neg*1.0)/(tot*1.0);
	        double child_attr_entropy = -1.0 * ( (p_pos*log(p_pos)) + (p_neg*log(p_neg)) );
	        if(num_neg == 0 || num_pos == 0)child_attr_entropy = 0;
	        weighted_av+= (-1.0) * ((tot*1.0)/(left.size()*1.0)*child_attr_entropy);
    	}

    	//cout << initial_entropy << " " << weighted_av <<  endl;
    	info_gain = initial_entropy + weighted_av ;
    	//cout << info_gain << endl;
    	return info_gain;
}

int getmaxgain(int sample_current[][no_attr], int output_current[no_attr], vector <int> left , int used[]){
    	double max_gain = 0;
    	int attr_no = -1;
    	for(int i = 0 ; i < no_attr ; i++)
	{
        	if(used[i])continue;
        	else
		{
            		double gain = information_gain(sample_current, output_current, left,i);
            		if(gain==1.0)return i;
            		if(gain > max_gain)
			{
                		max_gain = gain;
                		attr_no = i;
            		}
        	}
    	}
   	return attr_no;
}

int id3(int sample_current[][no_attr], int output_current[no_attr], node *cur_node){
    	vector <int> left(cur_node->left);
    	cnt_edges++;
    	int max_gain_attr = getmaxgain(sample_current, output_current, left,cur_node->used);
    	if(max_gain_attr == -1)
	{ // no more possible attributes to split
        	int count_one = 0 , count_zero = 0;
        	for(int i = 0 ; i < left.size() ; i++)
		{
            		int inp_ind = left[i];
            		if(output_current[inp_ind]==1)count_one++;
            		else count_zero++;
        	}
        	cur_node->attr_no = -1;
        	if(2*count_zero >= left.size())cur_node->out = 0;
        	else cur_node->out = 1;
        	return 0;
    	}
    	cur_node->attr_no = max_gain_attr;
    	cur_node->used[max_gain_attr] = 1;
    	for(int i = 0 ; i < attr_type[max_gain_attr].size() ; i++)
	{
        	node *temp = new node;
        	temp->child.clear();
        	temp->left.clear();
        	memset(temp->used,0,sizeof(temp->used));
        	for(int j = 0 ; j < no_attr ; j++)
		{
            		temp->used[j] = cur_node->used[j];
        	}
        	temp->out = -1;
        	for(int j =  0 ; j < left.size() ; j++)
		{
            		int inp_ind = left[j];
            		if(sample_current[inp_ind][max_gain_attr] == attr_type[max_gain_attr][i])
			{
                		temp->left.pb(inp_ind);
            		}
        	}
        	cur_node->child.pb(temp);
        	id3(sample_current, output_current, temp);
    	}
    	return 0;
}

int getAccProbability()
{
	double cumWeights[T] = {0};
	int index = T-1;
	cumWeights[0] = weights[0];
	for(int i=1; i<T; i++)
	{
		cumWeights[i] = cumWeights[i-1]+weights[i];
	}
	double random = (rand()%100)/100.0;
	while(index>=0)
	{
		if(cumWeights[index]<random)
			break;
		index--;
	}
	return (index+1)%T;
}

//to get the output for an example for a given classifier
int getOutput(int inp[], node* root)
{
	int val=root->attr_no;
	if(val==-1)
	{
	        return root->out;
	}

	return getOutput(inp, root->child[inp[val]]);
}


int formClassifier(int classifier_num) 
{
	//srand(time(NULL));
	int count_neg = 0;
	double error = 0, weights_sum = 0;
	//return ;
	for(int i=0; i<num_samples; i++)
	{
		samples_training[i] = getAccProbability();
	}
	for(int i=0; i<num_samples; i++)
	{
		for(int j=0; j<no_attr; j++)
			samples_current[i][j] = training_examples[samples_training[i]][j];
		output_current[i] = training_output[samples_training[i]];
	}
	node classifier;
	init_root(&classifier);
	cnt_edges = 0;
	id3(samples_current, output_current, &classifier);
	classifiers[classifier_num] = classifier;
	for(int i=0; i<T; i++)
	{
		double temp = getOutput(training_examples[i],&classifiers[classifier_num]);
		if(temp!=training_output[i])
		{
			count_neg++;
			error+= weights[i];
		}
		weights_sum+=weights[i];
	}
	//cout << count_neg << endl;
	//cout << error << " " << weights_sum << endl;
	coeffs[classifier_num] = log((1.0-error)/error)/(2.0);
	//update weights
	double sum = 0;
	//cout << coeffs[classifier_num] << endl;
	for(int i=0; i<T; i++)
	{
		double temp = getOutput(training_examples[i],&classifiers[classifier_num]);
		if(temp==0) temp=-1;
		weights[i] = weights[i]*exp(-(training_output[i]==0?-1:1)*coeffs[classifier_num]*temp);
		
		sum+=weights[i];
	}
	//normalize weights
	for(int i=0; i<T; i++)
	{
		weights[i]/=sum;
	}
	if(coeffs[classifier_num]<=0)
	{
		return 1;
	}
	return 0;
}

//Adaptive Boosting function
void adaBoost()
{
	srand(time(NULL));
	for(int i=0; i<T; i++)
		weights[i] = 1/((double)T);
	init();
	//cout << 1.0/T << endl;
	for(int i=0; i<L; i++)
	{
	//	cout << i << ":" <<endl;
		if(formClassifier(i))i--;
	}
}

int getBoostedOutput(int example[])
{
	double outputs[L] = {0};
	for(int i=0; i<L; i++)
	{
		outputs[i] = getOutput(example, &classifiers[i]);
		if(outputs[i]==0) outputs[i] = -1;
	}
	double output = 0;
	for(int i=0; i<L; i++)
	{
		output+=outputs[i]*coeffs[i];
	}
	if(output<0)
		return 0;
	return 1;
}

//To get the accuracy of the boosted ensembler on a set of examples
void getAccuracy(int examples[][no_attr], int output[], int num)
{
	int count = 0;
	for(int i=0; i<num; i++)
	{
		int temp = getBoostedOutput(examples[i]);
		if(temp==output[i])
			count++;
	}
	cout << (count*100.0)/num << endl;
}


int main () //***done***
{
	//load training examples and their actual output in training_examples and training_output
	freopen("adultdiscdata.txt","r",stdin);
	for(int i=0;i<T;i++)
	{
        	for(int j=0;j<=no_attr;j++)
        	{
        		if(j==no_attr)
			{
                		scanf("%d",&training_output[i]);
			}
            		else
                		scanf("%d",&training_examples[i][j]);
        	}
    	}
	fclose(stdin);
	//load testing examples
	freopen("adulttestdiscdata.txt","r",stdin);
	for(int i=0;i<Tst;i++)
	{
	        for(int j=0;j<=no_attr;j++)
	        {
        		if(j==no_attr)
		        {
                		scanf("%d",&testing_output[i]);
			}
            		else
                		scanf("%d",&testing_examples[i][j]);
        	}
    	}
	fclose(stdin);
	//adaptive boost function call
	clock_t begin = clock();
	adaBoost();
	clock_t end = clock();
	cout << "The training set accuracy is: ";
	getAccuracy(training_examples, training_output, T);
	cout << "The testing set accuracy is: ";
	getAccuracy(testing_examples, testing_output, Tst);
	cout << "Time takes in seconds: " << double(end - begin) / CLOCKS_PER_SEC << endl;


return 0;
}
