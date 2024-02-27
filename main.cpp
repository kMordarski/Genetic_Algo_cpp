#include <iostream>
#include "class.h"

int main() {
    std::string filename = "Returns_data1.csv";
    size_t populationSize = 120;
    size_t generations = 450;
    size_t tournamentSize = 80;

    std::vector<StockData> stockData = readCSV(filename);
    std::vector<std::vector<double>> covarianceMatrix = calculateCovarianceMatrix(stockData);

    // Assume you have an initial set of weights
    std::vector<double> initialWeights = createPortfolioWeights(stockData);
    normalizeVector(initialWeights);
    
    GeneticAlgorithm geneticAlgorithm(initialWeights, populationSize, generations, stockData, covarianceMatrix);
    geneticAlgorithm.runGeneticAlgorithm(tournamentSize);

    geneticAlgorithm.printTopWeights(initialWeights, stockData);

    system("pause");
    
    return 0;
}
