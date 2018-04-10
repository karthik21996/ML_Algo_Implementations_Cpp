#include <bits/stdc++.h>
using namespace std;
const int data_size=451,n=70,m=60,test_size=150;
int tr_out[data_size+1],no_pos,no_neg,test_out[test_size+1];
string tr_inp[data_size][n+1],test_inp[test_size][n+1];
double prob_face,prob_no_face;
double fgy[n+1][m+1],fgn[n+1][m+1];
double scale=1.5,smooth=1;
string s;
double conf_mat[2][2],tn[2][2];
int getaccuracytrain()
{
    int ctr=0;
    for(int i=0;i<data_size;i++)
    {
        double pp=prob_face,pn=prob_no_face;
        int out=0;
        for(int j=0;j<n;j++)
            for(int k=0;k<m;k++)
            {
                //cout<<pp<<" "<<pn<<endl;
                if(tr_inp[i][j][k]=='#')
                {
                    pp*=fgy[j][k];
                    pn*=fgn[j][k];
                }
                else
                {
                    pp*=(1-fgy[j][k]);
                    pn*=(1-fgn[j][k]);
                }
                pp*=scale;
                pn*=scale;
            }
        //cout<<pp<<" "<<pn<<endl;
        if(pp>=pn) out=1;
        //cout<<out<<endl;
        if(out==tr_out[i]) ctr++;
    }
    double acc=(ctr*100.0)/(data_size*1.0);
    cout<<"The training set accuracy is " << fixed<<setprecision(6)<<acc<<endl;
}
int getaccuracytest()
{
    int ctr=0;
    for(int i=0;i<test_size;i++)
    {
        double pp=prob_face,pn=prob_no_face;
        int out=0;
        for(int j=0;j<n;j++){
            for(int k=0;k<m;k++)
            {
                if(test_inp[i][j][k]=='#')
                {
                    pp*=fgy[j][k];
                    pn*=fgn[j][k];
                }
                else
                {
                    pp*=(1-fgy[j][k]);
                    pn*=(1-fgn[j][k]);
                }
                pp*=scale;
                pn*=scale;
            }
            //cout<<i<<" "<<j<<endl;
        }
        //cout<<pp<<" "<<pn<<endl;
        if(pp>=pn) out=1;
        //cout<<out<<endl;
        if(out==test_out[i]){ ctr++;
            if(out==1) conf_mat[0][0]++;
            else conf_mat[0][1]++;
        }
        else
        {
        	//cout<<out<<" "<<i<<endl;
            if(out==1) conf_mat[1][0]++;
            else conf_mat[1][1]++;
        }
    }
    double acc=(ctr*100.0)/(test_size*1.0);
    cout<< "The testing set accuracy is " << fixed<<setprecision(6)<<acc<<endl;
}
int main()
{
    freopen("facedatatrainlabels","r",stdin);
    for(int i=0;i<data_size;i++)
    {
        scanf("%d",&tr_out[i]);
        if(tr_out[i]==1) prob_face++,no_pos++;
        else prob_no_face++,no_neg++;
    }
    prob_face=prob_face/(data_size*1.0);
    prob_no_face=prob_no_face/(data_size*1.0);
    fclose(stdin);
    freopen("facedatatrain","r",stdin);
    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
        {
            fgy[i][j]=smooth;
            fgn[i][j]=smooth;
        }
    for(int i=0;i<data_size;i++)
    {
        for(int j=0;j<n;j++)
        {
            getline(cin,s);
            tr_inp[i][j]=s;
            for(int k=0;k<s.size();k++)
            {
                if(tr_out[i]==1 && s[k]=='#')
                    fgy[j][k]++;
                if(tr_out[i]==0 && s[k]=='#')
                    fgn[j][k]++;
            }
        }
    }
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++)
        {
            fgy[i][j]/=(no_pos*1.0+2*smooth);
            fgn[i][j]/=(no_neg*1.0+2*smooth);
            //cout<<fgy[i][j]<<" ";
        }
        //cout<<endl;
    }
    fclose(stdin);
    freopen("facedatatestlabels","r",stdin);
    for(int i=0;i<test_size;i++)
        scanf("%d",&test_out[i]);
    fclose(stdin);
    freopen("facedatatest","r",stdin);
    for(int i=0;i<test_size;i++)
    {
        for(int j=0;j<n;j++)
        {
            getline(cin,s);
            test_inp[i][j]=s;
        }
    }
    getaccuracytrain();
    getaccuracytest();
    
    for(int i=0;i<2;i++)
        for(int j=0;j<2;j++)
            tn[i][j]=conf_mat[i][j];
    conf_mat[0][0]/=(tn[0][0]+tn[1][0]);
    conf_mat[1][0]/=(tn[0][0]+tn[1][0]);
    conf_mat[0][1]/=(tn[0][1]+tn[1][1]);
    conf_mat[1][1]/=(tn[0][1]+tn[1][1]);
    cout << "\n\nClassification Matrix\n\n" ;
    for(int i=0;i<2;i++)
    {
        for(int j=0;j<2;j++)
            cout<<tn[i][j]<<" ";
        cout<<endl;
    }
    cout<<"\n\nConfusion matrix\n";
    for(int i=0;i<2;i++)
    {
        for(int j=0;j<2;j++)
            cout<<conf_mat[i][j]*100<<" ";
        cout<<endl;
    }
}
