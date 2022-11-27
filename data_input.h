//
// Created by jleyv on 11/16/2022.
//

#ifndef DEEPLEARNINGLIBRARY_DATA_INPUT_H
#define DEEPLEARNINGLIBRARY_DATA_INPUT_H

#include<iostream>
#include<vector>

class dataframe{
    public:
        std::vector<std::vector<double>> data;
        std::vector<int> labels;

        dataframe getSample(double frac);

        std::vector<std::vector<double>> getData(){return data;}

        std::vector<int> getLabels(){return labels;}

        dataframe(std::vector<std::vector<double>> data, std::vector<int> labels) : data(data), labels(labels){}
};

class data_input{
    public:
        std::string fileName;
        std::string delimeter;

        // Function to fetch data from a CSV File
        dataframe getData(int col_index_of_labels);

        bool isDouble(std::string const &str);

        bool isInteger(std::string const &str);

        data_input(std::string filename, std::string delm = ",") : fileName(filename), delimeter(delm){}
};


#endif //DEEPLEARNINGLIBRARY_DATA_INPUT_H
