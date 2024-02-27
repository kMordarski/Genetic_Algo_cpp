#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <unordered_map>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <numeric>

#include "measures.h"

// Function to create a vector of weights for the portfolio
std::vector<double> createPortfolioWeights(const std::vector<StockData>& stockData) {
    size_t numStocks = stockData.size();
    std::vector<double> weights; // Initialize with equal weights

    std::srand(std::time(0));

    for (size_t i = 0; i < stockData.size(); ++i) {
        double randomValue = static_cast<double>(std::rand()) / RAND_MAX;  // Random value between 0 and 1
        weights.push_back(randomValue);
    }

    
    return weights;
}

void normalizeVector(std::vector<double>& vector) {
    // Calculate the sum of the vector elements
    double sum = std::accumulate(vector.begin(), vector.end(), 0.0);

    // Normalize each element by dividing it by the sum
    for (size_t i = 0; i < vector.size(); ++i) {
        vector[i] /= sum;
    }
}

class GeneticAlgorithm {
public:
    GeneticAlgorithm(const std::vector<double>& initialWeights, size_t populationSize, size_t generations,
                     const std::vector<StockData>& stockData, const std::vector<std::vector<double>>& covarianceMatrix)
      : initialWeights_(initialWeights), populationSize_(populationSize), generations_(generations),
          stockData_(stockData), covarianceMatrix_(covarianceMatrix) {
        // Initialize random seed
        std::srand(static_cast<unsigned int>(std::time(0)));
    }

    void printTopWeights(const std::vector<double>& weights, const std::vector<StockData>& stockData) const {
        // Create a vector of pairs (weight, stock name)
        std::vector<std::pair<double, std::string>> weightStockPairs;

        for (size_t i = 0; i < weights.size(); ++i) {
            weightStockPairs.emplace_back(weights[i], stockData[i].symbol);
        }

        // Sort the vector based on weights in descending order
        std::sort(weightStockPairs.rbegin(), weightStockPairs.rend());

        // Print the top 10 weights and corresponding stock names
        std::cout << "Top 10 weights and corresponding stocks:\n";
        for (size_t i = 0; i < std::min<size_t>(10, weightStockPairs.size()); ++i) {
            std::cout << "Asset " << i + 1 << ": " << weightStockPairs[i].second << " - Weight: " << weightStockPairs[i].first << "\n";
        }
    }
  
    double runGeneticAlgorithm(size_t tournamentSize) {
        initializePopulation();
	double result;
        for (size_t generation = 0; generation < generations_; ++generation) {
	    calculateFitness();
            performTournamentSelection(tournamentSize);
r            performCrossover();
            performMutation();
        }

	// std::vector<std::vector<double>> covarianceMatrix = calculateCovarianceMatrix(stockData);
        // After running the algorithm, the best weights should be in the initialWeights_ vector
        std::cout << "Best weights after " << generations_ << " generations: ";
        printWeights(initialWeights_);

        // Calculate Sharpe ratio for the best weights
        double sharpeRatio = calculateSharpeRatio(initialWeights_, stockData_, covarianceMatrix_);
        std::cout << "Sharpe Ratio for the best weights: " << fitnessValues_.back() << std::endl;

	std::cout << "The sum of the weights is:" << std::accumulate(initialWeights_.begin(), initialWeights_.end(), 0.0) << std::endl;

	return result;
    }

private:
    std::vector<std::vector<double>> population_;
    std::vector<double> fitnessValues_;
    std::vector<double> initialWeights_;
    size_t populationSize_;
    size_t generations_;
    size_t tournamentSize_;
    
  
    const std::vector<StockData>& stockData_;
    const std::vector<std::vector<double>>& covarianceMatrix_;

    std::vector<double> initializeLHSWeights() const {
        std::vector<double> lhsWeights(initialWeights_.size());

        // Generate a Latin Hypercube sample
        LatinHypercubeSample(lhsWeights);

        // Normalize the weights
        normalizeWeights(lhsWeights);

        return lhsWeights;
    }

    void performTournamentSelection(size_t tournamentSize) {
        std::vector<std::vector<double>> selectedPopulation;

        for (size_t i = 0; i < populationSize_; ++i) {
            // Randomly choose tournamentSize individuals
            std::vector<size_t> tournamentIndices;
            for (size_t j = 0; j < tournamentSize; ++j) {
                tournamentIndices.push_back(rand() % populationSize_);
            }
 
            // Find the individual with the highest fitness in the tournament
            auto maxFitnessIt = std::max_element(tournamentIndices.begin(), tournamentIndices.end(), [this](size_t index1, size_t index2) { return fitnessValues_[index1] < fitnessValues_[index2];});

             // Add the selected individual to the new population
             selectedPopulation.push_back(population_[*maxFitnessIt]);
        }

        // Update the population with the selected individuals
        population_ = selectedPopulation;
    }
  
   
    void LatinHypercubeSample(std::vector<double>& sample) const {
        size_t numSamples = sample.size();
        std::vector<std::vector<double>> lhsMatrix(numSamples, std::vector<double>(numSamples, 0.0));

        // Fill the LHS matrix with random samples in each row
        for (size_t i = 0; i < numSamples; ++i) {
            for (size_t j = 0; j < numSamples; ++j) {
                lhsMatrix[i][j] = static_cast<double>(rand()) / RAND_MAX;
           }
        }

        // Shuffle each column of the matrix
        for (size_t i = 0; i < numSamples; ++i) {
             std::random_shuffle(lhsMatrix[i].begin(), lhsMatrix[i].end());
        }

         // Assign the LHS sample from the matrix
        for (size_t i = 0; i < numSamples; ++i) {
             sample[i] = lhsMatrix[i][rand() % numSamples];
        }
    }
  
    void initializePopulation() {
        population_.clear();
        fitnessValues_.clear();

        //for (size_t i = 0; i < populationSize_; ++i) {
        //    std::vector<double> randomWeights = generateRandomWeights();
        //    normalizeWeights(randomWeights);
        //    population_.push_back(randomWeights);
	// }
        for (size_t i = 0; i < populationSize_; ++i) {
            std::vector<double> lhsWeights = initializeLHSWeights();
            population_.push_back(lhsWeights);
        }
    }

    void calculateFitness() {
        fitnessValues_.clear();

	double fitness;
	
        for (const auto& weights : population_) {

	    // Normalize weights
	    std::vector<double> normalizedWeights = weights;
	    normalizeWeights(normalizedWeights);
	    
            fitness = calculateSharpeRatio(weights, stockData_, covarianceMatrix_);
            fitnessValues_.push_back(fitness);
	    // Print the fitness value for debugging
	    
        }
	std::cout << "Fitness: " << fitness << std::endl;
    }
  
    void performCrossover() {
         for (size_t i = 0; i < populationSize_; i += 2) {
             // Increase the crossover probability
             if (static_cast<double>(rand()) / RAND_MAX < 0.8) {
                size_t crossoverPoint = rand() % initialWeights_.size();
                for (size_t j = crossoverPoint; j < initialWeights_.size(); ++j) {
                     std::swap(population_[i][j], population_[i + 1][j]);
                }
                normalizeWeights(population_[i]);
                normalizeWeights(population_[i + 1]);
            }
        }
    }

    void performMutation() {
        for (auto& weights : population_) {
            for (size_t i = 0; i < weights.size(); ++i) {
                if (rand() % 100 < 10) { // 5% chance of mutation
                    weights[i] += (rand() % 11 - 5) / 100.0;
                }
            }
            normalizeWeights(weights);
        }
    }

    std::vector<double> generateRandomWeights() const {
        std::vector<double> randomWeights;
        for (size_t i = 0; i < initialWeights_.size(); ++i) {
            double randomWeight = static_cast<double>(rand()) / RAND_MAX;
            randomWeights.push_back(randomWeight);
        }
        return randomWeights;
    }

    void normalizeWeights(std::vector<double>& weights) const {
        double maxWeight = *std::max_element(weights.begin(), weights.end());
        for (auto& weight : weights) {
            weight /= maxWeight;
        }
    }

    void printWeights(const std::vector<double>& weights) const {
        for (size_t i = 0; i < weights.size(); ++i) {
            std::cout << "Asset " << i + 1 << ": " << weights[i] << " ";
        }
        std::cout << std::endl;
    }

    // Function to calculate the Sharpe ratio for a given portfolio
double calculateSharpeRatio(const std::vector<double>& weights, const std::vector<StockData>& stockData, const std::vector<std::vector<double>>& covarianceMatrix) {
    // Calculate the portfolio return
    double portfolioReturn = 0.0;
    for (size_t i = 0; i < stockData.size(); ++i) {
         portfolioReturn += weights[i] * stockData[i].expectedReturn;
    }

    double portfolioVolatility = 0.0;
    for (size_t i = 0; i < stockData.size(); ++i) {
        for (size_t j = 0; j < stockData.size(); ++j) {
            portfolioVolatility += weights[i] * weights[j] * covarianceMatrix[i][j];
        }
     }
    portfolioVolatility = std::sqrt(portfolioVolatility);

    // Calculate the Sharpe ratio
    if (portfolioVolatility != 0.0) {
        return (portfolioReturn) / portfolioVolatility;
    } else {
        return 0.0; // Handle the case where portfolio volatility is zero
    }
}
 
};
