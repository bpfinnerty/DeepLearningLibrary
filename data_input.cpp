//
// Created by jleyv on 11/16/2022.
//

#include <fstream>
#include "data_input.h"
#include <cmath>
#include <random>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

dataframe dataframe::getSample(double frac){
    std::vector<std::vector<double>> sample_data;
    std::vector<int> sample_labels;
    int num_of_entries = data.size();
    int num_of_samples = std::ceil(num_of_entries * frac);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, num_of_entries);

    for(int i = 0; i < num_of_samples; i++){
        auto index = distr(gen);
        sample_data.push_back(data[index]);
        sample_labels.push_back(labels[index]);
    }

    dataframe ret(sample_data, sample_labels);
}

bool data_input::isInteger(std::string const &str){
    try {
        boost::lexical_cast<int>(str);
        return true;
    }
    catch(boost::bad_lexical_cast &e) {
        return false;
    }
}

bool data_input::isDouble(std::string const &str)
{
    try {
        boost::lexical_cast<double>(str);
        return true;
    }
    catch(boost::bad_lexical_cast& e) {
        return false;
    }
}

dataframe data_input::getData(int col_index_of_label) {
    std::ifstream file(fileName);
    std::vector<std::vector<double>> dataList;
    std::vector<int> labels;

    std::string line;
    int expectedSizeOfInput = 0;
    bool foundSizeOfInput = false;

    // Read file line by line
    while (getline(file, line)){
        std::vector<std::string> vec;
        std::vector<double> vec_data;
        boost::algorithm::split(vec, line, boost::is_any_of(delimeter));

        // Skip the first line if it does not contain a number, as this is most likely the header info
        if(!data_input::isInteger(vec.at(0))){
            continue;
        }

        // go through the parsed vector to convert from strings to numbers
        for(int i = 0; i < vec.size(); i++){
            if(i == col_index_of_label){
                // if we reach the index that contains the label information, place in label vector

                if(isInteger(vec.at(i))){
                    int label = boost::lexical_cast<int>(vec.at(i));
                    labels.push_back(label);
                }else {
                    std::cout << "got invalid data\n";
                    exit(0);
                }
            }else{
                // else we just convert string to a double

                if(isDouble(vec.at(i))){
                    double val = boost::lexical_cast<double>(vec.at(i));
                    vec_data.push_back(val);
                }else{
                    std::cout << "got invalid data\n";
                    exit(0);
                }
            }
        }

        if(!foundSizeOfInput){
            // Check to see what the expected size of the input should be and store that for later checks
            expectedSizeOfInput = vec.size();
            foundSizeOfInput = true;
        }else{
            // If the next line does not have the expected number of inputs,
            if(vec.size() != expectedSizeOfInput){
                std::cout << "Error: expected data input size of " << expectedSizeOfInput << ", got " << vec.size();
                exit(0);
            }
        }
        dataList.push_back(vec_data);
    }

    file.close();

    dataframe ret(dataList, labels);

    return ret;
}
