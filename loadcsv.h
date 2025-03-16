#pragma once

#include <vector>
#include <string>

struct Data {
    std::vector<std::string> header;
    std::vector<std::vector<double>> X_train;
    std::vector<double> Y_train;
    std::vector<std::vector<double>> X_test;
    std::vector<double> Y_test;
    
};

Data load_csv(const std::string& filename, const std::string& target_column, double training_ratio);

