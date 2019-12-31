#include <iostream>
#include <vector>
#include <tuple>
#include <stdio.h>
#include <algorithm>
#include <iterator>
using namespace std;
// void getSequence(vector<vector<int>>& result_n){
//     vector<int> a = {1,1,1,1};
//     vector<int> b = {2,2,2,2};
//     result_n.push_back(a);
//     result_n.push_back(b);

// }

int main(){
    // int count_n=3;
    // for (size_t i = 0; i < count_n; i++)
    // {
    //     vector<vector<int>> result_n;
    //     getSequence(result_n);
        
    // }
    
    // vector<int> b = { 24,25};
    // vector<int> a ={1,2,3,24,25};
    // // for (size_t a : b)
    // // {
    // //     printf("\t%d", a);

    // // }
    
    // for (size_t i :b)
    // {
       
    //     a.erase(remove(a.begin(),a.end(),i),a.end());
    // }
    // for(vector<int>::iterator it =a.begin();it!=a.end();++it)
    // {
    //     cout<<*it;
    // }
    float total_score[4][2] ={0.0f};
    total_score[2][0] = 5;
    total_score[2][1] = 6;
    total_score[3][0] = 15;
    total_score[3][1] = 16;
    for (size_t i = 0; i <4; i++)
    {
       for (size_t j = 0; j < 2; j++)
       {
           cout<<total_score[i][j]<<endl;
       }
       
    }
    
    
}