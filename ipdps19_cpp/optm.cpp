#include "optm.h"
#include "global.h"
#include <omp.h>
#include <algorithm>
#include <utility>
#include <math.h>
#include <assert.h>

ADAM::ADAM(){lr = -1;}

ADAM::ADAM(std::vector<std::pair<int,int>> dims2d, std::vector<int> dims1d, double lr_)
{
    lr=lr_<=0 ? ADAM_LR:lr_;
    beta1=ADAM_BETA1;
    beta2=ADAM_BETA2;
    epsilon=ADAM_EPSILON;
    t=0;
    for (auto itr=dims2d.begin(); itr!=dims2d.end(); itr++) {
        ss2d.push_back(s_data2d_ds(itr->first,itr->second));
        rs2d.push_back(s_data2d_ds(itr->first,itr->second));
    }
    for (auto itr1=dims1d.begin(); itr1!=dims1d.end(); itr1++) {
        ss1d.push_back(s_data1d_ds(*itr1));
        rs1d.push_back(s_data1d_ds(*itr1));
    }
}


void ADAM::update(std::vector<s_data2d_ds*> weights, std::vector<s_data2d_ds*> d_weights,
                  std::vector<s_data1d_ds*> biases, std::vector<s_data1d_ds*> d_biases)
{
    // asertions
    for (int j=0; j<weights.size(); j++) {
        assert(weights[j]->dim1 == d_weights[j]->dim1);
        assert(weights[j]->dim2 == d_weights[j]->dim2);
        assert(weights[j]->dim1 == ss2d[j].dim1);
        assert(weights[j]->dim2 == ss2d[j].dim2);
    }
    t+=1;
    double denom1 = 1-powf(beta1,t);
    double denom2 = 1-powf(beta2,t);
    // update 2d parameters
    for (int i=0; i<weights.size(); i++) {
        int tot_elements = weights[i]->dim1*weights[i]->dim2;
        #pragma omp parallel for
        for (int k=0; k<tot_elements; k++) {
            ss2d[i].arr[k] = beta1*ss2d[i].arr[k]+(1-beta1)*d_weights[i]->arr[k];
            rs2d[i].arr[k] = beta2*rs2d[i].arr[k]+(1-beta2)*d_weights[i]->arr[k]*d_weights[i]->arr[k];
            weights[i]->arr[k] -= lr*powf(denom2,0.5)/denom1*ss2d[i].arr[k]/(powf(rs2d[i].arr[k],0.5)+epsilon);
        }
    }
    // update 1d parameters
    for (int i=0; i<biases.size(); i++) {
        int tot_elements = biases[i]->dim1;
        #pragma omp parallel for
        for (int k=0; k<tot_elements; k++) {
            ss1d[i].arr[k] = beta1*ss1d[i].arr[k]+(1-beta1)*d_biases[i]->arr[k];
            rs1d[i].arr[k] = beta2*rs1d[i].arr[k]+(1-beta2)*d_biases[i]->arr[k]*d_biases[i]->arr[k];
            biases[i]->arr[k] -= lr*powf(denom2,0.5)/denom1*ss1d[i].arr[k]/(powf(rs1d[i].arr[k],0.5)+epsilon);
        }
    }
}


