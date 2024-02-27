#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <unordered_map>


struct StockData {
    std::string symbol;
    std::vector<double> returns;
    double expectedReturn;
    // Add other relevant fields as needed
};

// Function to calculate the mean (expected return) of a vector of values
double calculateMean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0; // Return 0 if the vector is empty (handle this case as needed)
    }

    double sum = 0.0;
    for (const auto& value : values) {
        sum += value;
    }

    return sum / values.size();
}

// Function to calculate the variance-covariance matrix including all stocks
std::vector<std::vector<double>> calculateCovarianceMatrix(const std::vector<StockData>& stockData) {
    size_t numStocks = stockData.size();

    // Initialize a matrix filled with zeros
    std::vector<std::vector<double>> covarianceMatrix(numStocks, std::vector<double>(numStocks, 0.0));

    // Calculate the variance-covariance matrix for all stocks
    for (size_t i = 0; i < numStocks; ++i) {
        for (size_t j = 0; j < numStocks; ++j) {
            // Calculate the covariance between stock i and stock j
            double covariance = 0.0;
            for (size_t k = 0; k < stockData[i].returns.size(); ++k) {
                covariance += stockData[i].returns[k] * stockData[j].returns[k];
            }
            covariance /= (stockData[i].returns.size() - 1); // Adjust for sample size

            covarianceMatrix[i][j] = covariance;
        }
    }

    return covarianceMatrix;
}

std::vector<StockData> readCSV(const std::string& filename) {
    std::vector<StockData> stocks;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return stocks; // Return an empty vector on error
    }

    // Read the header row to get the stock symbols
    std::string header;
    if (std::getline(file, header)) {
        std::istringstream headerStream(header);
        std::string token;

        // Read the rest of the tokens as stock symbols directly
        // and initialize each StockData entry with its symbol
        while (std::getline(headerStream, token, ',')) {
            StockData stock;
            stock.symbol = token;
            stocks.push_back(stock);
        }

        // Read data for each stock
        while (std::getline(file, header)) {
            std::istringstream dataStream(header);

            // Read the first token as the stock name
            size_t index = 0;
            while (std::getline(dataStream, token, ',')) {
                try {
                    double returnVal = std::stod(token);
                    stocks[index].returns.push_back(returnVal);
                    index++;
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Error converting return value to double on line: " << header << std::endl;
                }
            }
        }

        // Calculate expected returns for each stock
        for (auto& stock : stocks) {
            stock.expectedReturn = calculateMean(stock.returns);
        }
    } else {
        std::cerr << "Error reading header row from file: " << filename << std::endl;
    }

    file.close();

    return stocks;
}

// Function to calculate the Sharpe ratio for a given portfolio
double calculateSharpeRatio(const std::vector<double>& weights, const std::vector<StockData>& stockData, const std::vector<std::vector<double>>& covarianceMatrix, const double riskFreeRate) {
    // Calculate the portfolio return
    double portfolioReturn = 0.0;
    for (size_t i = 0; i < stockData.size(); ++i) {
        portfolioReturn += weights[i] * stockData[i].expectedReturn;
    }

    // Calculate the portfolio volatility (standard deviation)
    double portfolioVolatility = 0.0;
    for (size_t i = 0; i < stockData.size(); ++i) {
        for (size_t j = 0; j < stockData.size(); ++j) {
            portfolioVolatility += weights[i] * weights[j] * covarianceMatrix[i][j];
        }
    }
    portfolioVolatility = std::sqrt(portfolioVolatility);

    // Calculate the Sharpe ratio
    if (portfolioVolatility != 0.0) {
        return (portfolioReturn - riskFreeRate) / portfolioVolatility;
    } else {
        return 0.0; // Handle the case where portfolio volatility is zero
    }
}
