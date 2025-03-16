#pragma once

#include <vector>
#include <random>
#include <iostream>
#include "loadcsv.h"
#include <bits/stdc++.h>

class Perceptron{
    public:

    enum ActivationFunction{
        STEP,
        NONE,
    };

private:

    //Data Members
    std::vector<double> weights {};
    double bias{};
    double learning_rate {0.01};
    int row_size {};
    ActivationFunction activation_function;
    std::vector<double> data_min {};
    std::vector<double> data_max {};

    //Weights Initiazation
    void initializeWeights(int size){
        weights.resize(size);
        std::mt19937 mt {std::random_device{}()};
        std::uniform_real_distribution<double> weights_range (-1,1); //weights generation range
    
        for (int i{0}; i<size; ++i){
            weights[i]=weights_range(mt);
        }
        bias=weights_range(mt);
    }
    
    //dot product of two vectors 
    template <typename T, typename U> //ive used templates because its my code ;) (i wanted it to use in some other context)
    double dot(const std::vector<T>& vec1, const std::vector<U>& vec2){
        double dot {};
        for (int i {0}; i<vec1.size(); ++i){
            dot+=vec1[i]*vec2[i];
        }
        return dot;
    }

    //step function
    int step(double weighted_sum){
        return weighted_sum>=0.0 ? 1 : 0;
    }
    
    //training info each epoch (classification)
    void verbose_classification(int epoch, int correct, int size){
        std::cout<<"Epoch: "<<epoch<<" Accuracy: "<<(correct/static_cast<double>(size))*100<<"%"<<"\n";
    }
    
    //training info each epoch (regression)
    void verbose_regression(int epoch, double loss, int size){
        std::cout<<"Epoch: "<<epoch<<" Loss: "<<std::sqrt(loss/size)<<"\n"; 
    }
    
    //data noramalizing 
    template <typename T>
    std::vector<std::vector<T>> normalize(std::vector<std::vector<T>> inp_vec){
        
        data_min.resize(inp_vec[0].size());
        data_max.resize(inp_vec[0].size());
    
        for (int i {0}; i<inp_vec[0].size(); ++i){
            std::vector<T> temp(inp_vec.size());
            for (int j {0}; j<inp_vec.size(); ++j){
                temp[j] = inp_vec[j][i];
            }

            //stored min and max values in data_min and data_max to use them in predict function
            auto min = data_min[i] = *std::min_element(temp.begin(),temp.end());
            auto max = data_max[i] = *std::max_element(temp.begin(),temp.end());
            
            //normalize
            for (int j {0}; j<inp_vec.size(); ++j){
                inp_vec[j][i] = (inp_vec[j][i]-min)/static_cast<double>(max-min);
            }
        }
    
        return inp_vec;
    }

public:
    Perceptron(ActivationFunction activation_func) : activation_function {activation_func}{}

    //training function 
    template <typename T, typename U>
    void train(std::vector<std::vector<T>>& features, std::vector<U>& label, int epochs, bool verbose){
        int feature_size {static_cast<int>(features[0].size())};
        row_size=feature_size;
        int data_size {static_cast<int>(features.size())};
    
        features = normalize(features); //noramlize weights before training (this perceptron goes crazy if i dont)
    
        initializeWeights(feature_size); //weights init
    
        for (int e {1}; e<=epochs; ++e){ //runs through whole code every epoch
            
            //correct predictions (if classification)
            int correct {};

            //sum of all error (label-y_hat)
            double loss {};
            
            //runs for evey row in data
            for (int i {0}; i<data_size; ++i){

                double weighted_sum {dot(features[i], weights)+bias};
    
                double y_hat {};

                //if activation function is STEP then y_hat is step 
                //function applied on weighted sum (classification)

                if (activation_function==STEP) {
                    y_hat = step(weighted_sum);
                    correct += (y_hat==label[i] ? 1 : 0);
                }
                else{
                    y_hat = weighted_sum;   //if no activation function (NONE) then y_hat will be raw weighted sum (regression)
                    loss += (label[i]-y_hat)*(label[i]-y_hat);
                } 
                double update { learning_rate*(label[i]-y_hat) };

                //weights update
                for (int j{0}; j<feature_size; ++j){
                    weights[j] += update*features[i][j];
                }
                bias += update;
            }
        
        //calls suitable verbose function based on activation function
        if(!(e%10)){
            if (verbose && activation_function==STEP) verbose_classification(e, correct, data_size);
            if (verbose && activation_function==NONE) verbose_regression(e, loss, data_size);
            }
        }
    }
    
    //predict function (ik its bit cluttered, whole code i mean :( )
    std::vector<double> predict(std::vector<std::vector<double>> to_pred){

            //size validation of no of columns in to_pred matrix 
            if (to_pred[0].size()!=row_size){
                throw std::invalid_argument("Invalid Input Vector Of Size: "+std::to_string(to_pred[0].size())
                +"\nExpected Size: "+std::to_string(row_size));
            }
            
            //normalize data before prediction 
            for (int i {0}; i<to_pred.size(); ++i){
                for (int j {0}; j<to_pred[0].size(); ++j){
                    to_pred[i][j] = (to_pred[i][j]-data_min[j])/static_cast<double>(data_max[j]-data_min[j]);
                }
            }
            
            //predictions matrix
            std::vector<double> predictions (to_pred.size());
            
            //predict
            for (int i {0}; i<to_pred.size(); ++i){
                if (activation_function==STEP) predictions[i] = step(dot(to_pred[i],weights)+bias);
                else predictions[i] = dot(to_pred[i],weights)+bias;
            }

            return predictions;
        }

    //some getters and setters
    const std::vector<double>& getWeights() {return weights;}
    void setWeights(const std::vector<double>& weights_vec) {weights=weights_vec;}
    void setLearningRate(double rate) {learning_rate=rate;}

};
