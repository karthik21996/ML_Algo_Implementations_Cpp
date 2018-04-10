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
int cnt_edges;
int tree_attr_no=7;
node root[104],mk[15000];
int tree_size = 12000;
int no_tree=40,valid_attr[20];
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
        if(valid_attr[i]==0) continue;
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
void getAccuracyTest()
{
    int cnt = 0;
    for(int i=0;i<test_size;i++){
        //cout << i << endl;
        int cz=0,co=0;
        for(int j=0;j<no_tree;j++){
            int val = getoutput(&root[j],test_inp[i]);
            if(val==1) co++;
            else cz++;
    }
    int ao=0;
    if(co>=cz) ao=1;
    if(ao==test_out[i])
        cnt++;
    }
    double acc = (cnt*100.0)/(test_size*1.0);
    cout <<"The accuracy on the testing set is " << fixed << setprecision(10) << acc << endl;
}
int random_forest()
{
    for(int i=0;i<no_tree;i++)
    {
        //cout<<i<<endl;
        root[i].left.resize(tree_size);
        root[i].out=-1;
        for(int j=0;j<14;j++) root[i].used[j]=0;
        for(int j=0;j<tree_size;j++){
            root[i].left[j]=rand()%data_size;;
        }
        memset(valid_attr,0,sizeof(valid_attr));
        for(int j=0;j<tree_attr_no;j++){
            int val = rand()%no_attr;
            //cout << val << " ";
            valid_attr[val]=1;
            //valid_attr[j] = 1;
        }
        //cout << endl;
        id3(&root[i],-1000);
    }
}
int main()
{
    init();

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
    random_forest();
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    fclose(stdin);
    //getAccuracy();
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
    cout << "The number of trees in the random forest is " << no_tree << endl;
    cout << "The number of attributes in each tree is " << tree_attr_no << endl;
    cout << "The number of training set each tree is trained on " << tree_size << endl;
    getAccuracyTest();
    cout << "The time taken to run the Random Forest algorithm " << time_spent << endl;
}