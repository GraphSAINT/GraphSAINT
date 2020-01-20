#pragma once
#include "global.h"
#include <algorithm>
#include <vector>
#include <utility>

class ADAM
{
public:
    ADAM();
    ADAM(std::vector<std::pair<int,int>> dims2d, std::vector<int> dims1d, double lr_=0);
    void init(std::vector<std::pair<int,int>> dims2d, std::vector<int> dims1d);
    void update(std::vector<s_data2d_ds*> weights, std::vector<s_data2d_ds*> d_weights, 
                std::vector<s_data1d_ds*> biases, std::vector<s_data1d_ds*> d_biases);
private:
    double lr;
    double beta1;
    double beta2;
    double epsilon;
    std::vector<s_data2d_ds> ss2d;
    std::vector<s_data1d_ds> ss1d;
    std::vector<s_data2d_ds> rs2d;
    std::vector<s_data1d_ds> rs1d;
    t_idx t;
};

