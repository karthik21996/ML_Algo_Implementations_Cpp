#include <bits/stdc++.h>
#define ff first
#define ss second
#define mp make_pair
#define pb push_back
using namespace std;
const int data_size=32561,no_attr=14 , test_size = 16281;
typedef struct node
{
    vector<node*> child;
    vector<int> left;
    int used[20];
    int attr_no;
    int out;
}node;
vector<double> tr_inp[data_size+1],test_inp[test_size+1];
vector<int> attr_type[no_attr];
node root;
int cnt_edges;
double tr_out[data_size+1],test_out[test_size+1];
void init(){
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

void init_root(){
    root.left.resize(data_size);
    root.out=-1;
    for(int i=0;i<14;i++) root.used[i]=0;
    for(int i=0;i<data_size;i++) root.left[i]=i;
}
double information_gain(vector <int> left , int attr){
    double info_gain = 0;
    int num_pos = 0 , num_neg = 0;
    for(int i = 0 ; i < left.size() ; i++){
        int inp_ind = left[i];
        if(tr_out[inp_ind]==1)num_pos++;
        else num_neg++;
    }
    double p_pos = (num_pos*1.0)/(left.size()*1.0);
    double p_neg = (num_neg*1.0)/(left.size()*1.0);
    double initial_entropy = -1.0 * ( (p_pos*log(p_pos)) + (p_neg*log(p_neg)) );
  //  cout << initial_entropy << endl;
    if(num_pos == 0 || num_neg == 0){
        initial_entropy = 0; //  pure subset , hence leaf node
    }
    // now calculate the weighted average entropy after split
    double weighted_av = 0;
    for(int i = 0 ; i < attr_type[attr].size() ; i++ ){
        num_pos = 0;
        num_neg = 0;
        for(int j = 0 ; j < left.size() ; j++){
        int inp_ind = left[j];
            if(tr_inp[inp_ind][attr]==attr_type[attr][i]){
                if(tr_out[inp_ind]==1)num_pos++;
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
    info_gain = initial_entropy + weighted_av ;
    return info_gain;
}
int getmaxgain(vector <int> left , int used[]){
    double max_gain = 0;
    int attr_no = -1;
    for(int i = 0 ; i < no_attr ; i++){
        if(used[i])continue;
        else{
            double gain = information_gain(left,i);
            if(gain==1.0)return i;
            if(gain > max_gain){
                max_gain = gain;
                attr_no = i;
            }
        }
    }
    return attr_no;
}
int id3(node *cur_node , int par){
    vector <int> left(cur_node->left);
   // if(left.size()==0)return 0;
    cnt_edges++;
    int max_gain_attr = getmaxgain(left,cur_node->used);
    if(max_gain_attr == -1){ // no more possible attributes to split
        int count_one = 0 , count_zero = 0;
        for(int i = 0 ; i < left.size() ; i++){
            int inp_ind = left[i];
            if(tr_out[inp_ind]==1)count_one++;
            else count_zero++;
        }
        cur_node->attr_no = -1;
        if(2*count_zero >= left.size())cur_node->out = 0;
        else cur_node->out = 1;
        return 0;
    }
    cur_node->attr_no = max_gain_attr;
    cur_node->used[max_gain_attr] = 1;
    for(int i = 0 ; i < attr_type[max_gain_attr].size() ; i++){
        node *temp = new node;
        temp->child.clear();
        temp->left.clear();
        memset(temp->used,0,sizeof(temp->used));
        for(int j = 0 ; j < no_attr ; j++){
            temp->used[j] = cur_node->used[j];
        }
        temp->out = -1;
        for(int j =  0 ; j < left.size() ; j++){
            int inp_ind = left[j];
            if(tr_inp[inp_ind][max_gain_attr] == attr_type[max_gain_attr][i]){
                temp->left.pb(inp_ind);
            }
        }
        cur_node->child.pb(temp);
        id3(temp,max_gain_attr);
    }
    return 0;
}
int getoutput(node *root , vector <double> inp){
    int attr = root->attr_no ;
    if(attr == -1){//leaf
        return root->out;
    }
    return getoutput(root->child[inp[attr]] , inp);
}
void getAccuracy(){
    int cnt = 0,co=0;
    for(int i=0;i<data_size;i++){
        int val = getoutput(&root,tr_inp[i]);
        if(val==tr_out[i])
            cnt++;
        if(val==1) co++;
    }
    double acc = (cnt*100.0)/(data_size*1.0);
    cout << "The training set accuracy is " << fixed << setprecision(10) << acc << endl;
}
void getAccuracyTest()
{
    int cnt = 0 , co = 0;
    for(int i=0;i<test_size;i++){
        int val = getoutput(&root,test_inp[i]);
        //cout<<val<<endl;
        if(val==test_out[i]){
            cnt++;
        }
        if(val==1) co++;
    }
    //  cout << co << " " << test_size - co << endl;
    double acc = (cnt*100.0)/(test_size*1.0);
    cout << "The testing set accuracy is " <<  fixed << setprecision(10) << acc << endl;
}
int main()
{

    init();
    init_root();
    freopen("adultdiscdata.txt","r",stdin);
    for(int i=0;i<data_size;i++)
    {
        tr_inp[i].resize(no_attr);
        for(int j=0;j<=no_attr;j++)
        {
            if(j==no_attr)
                scanf("%lf",&tr_out[i]);
            else{
                scanf("%lf",&tr_inp[i][j]);
 
            }
        }
    }
    clock_t begin = clock();
    id3(&root,-1000);
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    fclose(stdin);
    getAccuracy();
    freopen("adulttestdiscdata.txt","r",stdin);
    for(int i=0;i<test_size;i++)
    {
        test_inp[i].resize(no_attr);
        for(int j=0;j<=no_attr;j++)
        {
            if(j==no_attr)
                scanf("%lf",&test_out[i]);
            else
                scanf("%lf",&test_inp[i][j]);
        }
    }
    getAccuracyTest();
    cout << "The time taken to run the id-3 algorithm " << time_spent << endl;
}