#include "loadcsv.h"
#include "perceptron.h"

int main(){
    Perceptron p {Perceptron::STEP};

    Data data {load_csv("b_data.csv", "diagnosis", 0.8)};

    p.train(data.X_train, data.Y_train, 300, true);
    auto pred {p.predict(data.X_test)};

    std::cout<<std::endl;

    for(int i {}; i<data.Y_test.size(); ++i){
        std::cout<<"Actual: "<<data.Y_test[i]<<" Predicted: "<<pred[i]<<"\n";
    }
    return 0;
}
