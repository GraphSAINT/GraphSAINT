#include "global.h"
#include "sample.h"
#ifdef USE_MKL
#include "mkl.h"
#endif
#include <omp.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <set>
#include <iostream>
#include <map>
#include <cstring>
#include <cassert>

using namespace std;


void check_DB(vector<t_idx> &DB0,vector<t_idx> &DB1,vector<t_idx> &DB2,t_idx size, int thread_inter)
{
    if (DB0.capacity()<size)
    {
        printf("Thread %d doubling from %ld to %ld.\n",thread_inter,DB0.capacity(),DB0.capacity()*2);
        DB0.reserve(DB0.capacity()*2);
        DB1.reserve(DB1.capacity()*2);
        DB2.reserve(DB2.capacity()*2);
    }
    DB0.resize(size);
    DB1.resize(size);
    DB2.resize(size);
}

void sample_frontier(s_data2d_sp adj_train, s_idx1d_ds node_train, s_data2d_sp *&subgs, s_idx1d_ds *&subgs_v, int subgraph_size, int size_frontier) 
{
    double tt1=omp_get_wtime();
    printf("Sampling %d subgraphs.\n",num_thread);
    #pragma omp parallel num_threads(num_thread)
    {
        double t1=omp_get_wtime(),tpre;
        int inter_thread_id=omp_get_thread_num();
        unsigned int myseed=inter_thread_id;
        int avg_deg=adj_train.num_e/adj_train.num_v;
        avg_deg=(avg_deg>SAMPLE_CLIP)?SAMPLE_CLIP:avg_deg;
        int iteration=subgraph_size-size_frontier;
        //DBx: Dashboard line x, IAx: Index array line x
        vector<t_idx> DB0,DB1,DB2,IA0,IA1,IA2,IA3,IA4,nDB0,nDB1,nDB2;
        DB0.reserve(avg_deg*size_frontier*ETA);
        DB1.reserve(avg_deg*size_frontier*ETA);
        DB2.reserve(avg_deg*size_frontier*ETA);
        IA0.reserve(subgraph_size);
        IA1.reserve(subgraph_size);
        IA2.reserve(subgraph_size);
        IA3.reserve(subgraph_size);
        IA4.reserve(subgraph_size);
        IA0.resize(size_frontier);
        IA1.resize(size_frontier);
        IA2.resize(size_frontier);
        IA3.resize(size_frontier);
        t_idx *sama=(t_idx*)malloc(sizeof(t_idx)*(1+1));    // intra_thread = 1
        t_idx choose,neigh_v,newsize;
        set<t_idx> st;
        // #pragma omp for
        for (int i=0;i<size_frontier;i++)
        {
            IA3[i]=node_train.arr[rand_r(&myseed)%node_train.dim1];
            st.insert(IA3[i]);
            IA0[i]=adj_train.indptr[IA3[i]+1]-adj_train.indptr[IA3[i]];
            IA0[i]=(IA0[i]>SAMPLE_CLIP)?SAMPLE_CLIP:IA0[i];
            IA1[i]=1;
            IA2[i]=0;
        }
        // calculate prefix sum for IA0 and store in IA2 to compute the address for each frontier in DB
        IA2[0]=IA0[0];
        for (int i=1;i<size_frontier;i++)
            IA2[i]=IA2[i-1]+IA0[i];
        // now fill DB accordingly
        check_DB(DB0,DB1,DB2,IA2[size_frontier-1],inter_thread_id);
        for (int i=0;i<size_frontier;i++)
        {
            t_idx DB_start=(i==0)?0:IA2[i-1];
            t_idx DB_end=IA2[i];
            for (int j=DB_start;j<DB_end;j++)
            {
                DB0[j]=IA3[i];
                DB1[j]=(j==DB_start)?(j-DB_end):(j-DB_start);
                DB2[j]=i+1;
            }
        }
        tpre=omp_get_wtime();
        for (int itr=0;itr<iteration;itr++)
        {
            choose=-1;
            while (choose==-1)
            {
                t_idx tmp=rand_r(&myseed)%DB0.size();
                if (tmp<DB0.size())
                    if (DB0[tmp]!=-1)
                        choose=tmp;
            }
            choose=(DB1[choose]<0)?choose:(choose-DB1[choose]);
            neigh_v=(adj_train.indptr[DB0[choose]+1]-adj_train.indptr[DB0[choose]]!=0)?rand_r(&myseed)%(adj_train.indptr[DB0[choose]+1]-adj_train.indptr[DB0[choose]]):-1;
            if (neigh_v!=-1)
            {
                neigh_v=adj_train.indices[adj_train.indptr[DB0[choose]]+neigh_v];
                st.insert(neigh_v);
                IA1[DB2[choose]-1]=0;
                IA0[DB2[choose]-1]=0;
                for (int i=choose;i<choose-DB1[choose];i++)
                {
                    DB0[i]=-1;
                }
                newsize=adj_train.indptr[adj_train.indices[neigh_v]+1]-adj_train.indptr[adj_train.indices[neigh_v]];
                newsize=(newsize>SAMPLE_CLIP)?SAMPLE_CLIP:newsize;
            }
            else
            {
                newsize=0;
            }
            //shrink DB to remove sampled nodes, also shrink IA accordingly
            bool cond=DB0.size()+newsize>DB0.capacity();
            if (cond)
            {
                // compute prefix sum for the location in shrinked DB
                IA4.resize(IA0.size());
                IA4[0]=IA0[0];
                for (int i=1;i<IA0.size();i++)
                    IA4[i]=IA4[i-1]+IA0[i];
                nDB0.resize(IA4.back());
                nDB1.resize(IA4.back());
                nDB2.resize(IA4.back());
                IA2.assign(IA4.begin(),IA4.end());
                for (int i=0;i<IA0.size();i++)
                {
                    if (IA1[i]==0)
                        continue;
                    t_idx DB_start=(i==0)?0:IA4[i-1];
                    t_idx DB_end=IA4[i];
                    for (int j=DB_start;j<DB_end;j++)
                    {
                        nDB0[j]=IA3[i];
                        nDB1[j]=(j==DB_start)?(j-DB_end):(j-DB_start);
                        nDB2[j]=i+1;
                    }
                }
                // remap the index in DB2 by compute prefix of IA1 (new idx in IA)
                IA4.resize(IA1.size());
                IA4[0]=IA1[0];
                for (int i=1;i<IA1.size();i++)
                    IA4[i]=IA4[i-1]+IA1[i];
                DB0.assign(nDB0.begin(),nDB0.end());
                DB1.assign(nDB1.begin(),nDB1.end());
                DB2.assign(nDB2.begin(),nDB2.end());
                for (auto i=DB2.begin();i<DB2.end();i++)
                    *i=IA4[*i-1];
                t_idx curr=0;
                for (t_idx i=0;i<IA0.size();i++)
                {
                    if (IA0[i]!=0)
                    {
                        IA0[curr]=IA0[i];
                        IA1[curr]=IA1[i];
                        IA2[curr]=IA2[i];
                        IA3[curr]=IA3[i];
                        curr++;
                    }
                }
                IA0.resize(curr);
                IA1.resize(curr);
                IA2.resize(curr);
                IA3.resize(curr);
            }
            check_DB(DB0,DB1,DB2,newsize+DB0.size(),inter_thread_id);
            IA0.push_back(newsize);
            IA1.push_back(1);
            IA2.push_back(IA2.back()+IA0.back());
            IA3.push_back(neigh_v);
            t_idx DB_start=(*(IA2.end()-2));
            t_idx DB_end=IA2.back();
            for (int j=DB_start;j<DB_end;j++)
            {
                DB0[j]=IA3.back();
                DB1[j]=(j==DB_start)?(j-DB_end):(j-DB_start);
                DB2[j]=IA3.size();
            }
        }
        double tpost=omp_get_wtime();
        // st.insert(node_train.arr,node_train.arr+node_train.dim1);        
        //construct adj of subgraph base on the sampled nodes
        t_idx thread=inter_thread_id;
        if (!subgs_v[thread].arr) delete [] subgs_v[thread].arr;
        subgs_v[thread].dim1=st.size();
        subgs_v[thread].arr=new t_idx[st.size()];
        if (!subgs[thread].indices) delete [] subgs[thread].indices;
        if (!subgs[thread].arr) delete [] subgs[thread].arr;
        if (!subgs[thread].indptr) delete [] subgs[thread].indptr;
        subgs[thread].indptr=new t_idx[st.size()+1];
        vector<t_idx> indices_v;
        vector<t_data> arr_v;
        vector<t_idx> lut,lut_inv;
        lut.resize(st.size());
        lut_inv.resize(adj_train.num_v);
        int curr=0;
        for (set<int>::iterator it=st.begin();it!=st.end();it++)
        {
            lut[curr]=*it;
            lut_inv[*it]=curr;
            //copy the vertex of subgraph
            subgs_v[thread].arr[curr]=*it;
            curr++;
        }
        subgs[thread].indptr[0]=0;
        for (int i=0;i<st.size();i++)
        {
            int cnt=0;
            for (int j=adj_train.indptr[lut[i]];j<adj_train.indptr[lut[i]+1];j++)
            {
                if (st.find(adj_train.indices[j])!=st.end())
                {
                    cnt++;
                    indices_v.push_back(lut_inv[adj_train.indices[j]]);
                }
            }
            subgs[thread].indptr[i+1]=subgs[thread].indptr[i]+cnt;
            for (int j=0;j<cnt;j++) arr_v.push_back((t_data)1.0/cnt);
        }
        subgs[thread].num_v=st.size();
        subgs[thread].num_e=indices_v.size();
        subgs[thread].indices=new t_idx[indices_v.size()];
        memcpy(subgs[thread].indices,&indices_v[0],indices_v.size()*sizeof(t_idx));
        subgs[thread].arr=new t_data[arr_v.size()];
        memcpy(subgs[thread].arr,&arr_v[0],arr_v.size()*sizeof(t_data));
        double t2=omp_get_wtime();
        printf("thread %d finish in %dms while pre use %dms and post use %dms.\n",thread,(int)((t2-t1)*1000),(int)((tpre-t1)*1000),(int)((t2-tpost)*1000));
        // printf("\t num_e:%lu numv:%lu\n",st.size(),indices_v.size());
    }
    double tt2=omp_get_wtime();
    printf("Sampling: total time %.8lfs.\n",tt2-tt1);
    return;
}


