import numpy as np
import configparser
import matplotlib.pyplot as plt

# Loading and reading the config the parser 
def loadConfig():
    config = configparser.ConfigParser()
    config.read('config.conf')
    return config

# Loads the input data file into a string variable
# @dataFile -> takes file path as input, loaded from config.conf file
def loadDataFile(dataFile):
    data = ""
    with open(dataFile, 'r', encoding='utf-8') as inputFile:
        readLines = inputFile.readlines()

        samples, bins = readLines[0].strip().split(' ')

        for line in readLines:
            data += line
    return int(samples), int(bins), data

# Input file converted to X and Y numpy arrays
# @param: dataStr is the data file converted into string
# @param: binSize is the feature size of the input file
def numpyConversion(dataStr, binSize):
    cleanStr = dataStr.rstrip("\x1a")
    lines = cleanStr.strip().split('\n')[1:]

    x = []
    y = []

    for line in lines:
        values = list(map(float, line.split()))
        y.append(values[0])
        x.append(values[1:])

    y = np.array(y)
    x = np.array([row[:binSize] for row in x])
    
    y = y.astype(int)
    print(x.shape)

    return x, y

# Initializing weight function
# @param: input -> where input is the input dimension
def initializeWeights(input, featureSize):
    # np.random.seed(4)
    return np.random.rand(input, input, featureSize)

# this is the Euclidean Function just a different name, finds the minimum index 
# or finds the minimum distanced neuron based on an input vector
def bestMatchingNeuron(weight, inputX, grid):
    gridSpace = (grid, grid)
    distance = np.linalg.norm(weight - inputX, axis = 2)
    minDistanceNeuron = np.argmin(distance)

    minIndex = np.unravel_index(minDistanceNeuron, gridSpace) # unwraps into coordinates

    return minIndex

def gaussianFunctionWrap(minDistanceIndex, grid):
    # @ r - row of the grid
    # @ c - column of the grid
    r, c = np.indices((grid, grid))

    # Computeing distances along with wrapping sides and top to bottom
    delta_r = np.minimum(np.abs(r - minDistanceIndex[0]), grid - np.abs(r - minDistanceIndex[0])) # handling the side wrapping, row-wise
    delta_c = np.minimum(np.abs(c - minDistanceIndex[1]), grid - np.abs(c - minDistanceIndex[1])) # handling the top-bottom wrapping, column-wise 

    # Euclidean distance between row and column wise
    gDist = np.sqrt(delta_r**2 + delta_c**2)

    sigma = 1 / np.sqrt(2 * np.log(2)) #0.85 # 50 % width between two data points 

    # neighborhood function - calculates the influence from a neuron to its surrounding nodes. 
    neighbourhood = np.exp(-(gDist ** 2) / (2 * (sigma ** 2)))

    return neighbourhood

def trainingSOM(weightMatrix, inputX, grid, learningRate, maxIterations):
    errors = []

    for i in range(maxIterations):
        # Random data initialization
        randomIndex = np.random.randint(0, inputX.shape[0])
        dataUnit = inputX[randomIndex]

        # find minimum distance between a vector and all the other neuron
        bestIndex = bestMatchingNeuron(weightMatrix, dataUnit, grid)

        # radius of a neuron defined
        radius = grid / 2 * np.exp(-i / maxIterations)

        # find the influence between best matching neurons and its surrounding neurons. 
        neighbourhood = gaussianFunctionWrap(bestIndex, grid)

        decay = np.exp(-i / maxIterations)  # Learning rate decay
        
        # Update weights
        for j in range(grid):
            for k in range(grid):
                w = weightMatrix[j, k]

                distance = np.linalg.norm([j - bestIndex[0], k - bestIndex[1]]) # distance 

                # if the current distance is of the neuron is <= radius then updates the neuron weights.
                if distance <= radius:
                    influence = neighbourhood[j, k] * decay
                    w += learningRate * influence * (dataUnit - w)
                weightMatrix[j, k] = w
        
        # summarizes the quantization error each 100th iteration.
        if i % 100 == 0:
            error = quantizationError(weightMatrix, inputX, grid)
            errors.append(error)
            print(f"Iteration {i}/{maxIterations}, Quantization Error: {error:.4f}")

    return weightMatrix, errors

# Normalizes activation matrix to 0 to 1.
def normalizedConfidence(confidenceMatrix):
    minVal = np.min(confidenceMatrix)
    maxVal = np.max(confidenceMatrix)

    return (confidenceMatrix - minVal) / (maxVal - minVal)

# finds quantization error with respect to a single data unit and SOM trained weight matrix 
def quantizationError(weightMatrix, inputX, grid):
    total_error = 0.0

    for dataUnit in inputX:
        # finds min index
        minIndex = bestMatchingNeuron(weightMatrix, dataUnit, grid)
        # finds the error of vector x
        error = np.linalg.norm(weightMatrix[minIndex] - dataUnit)
        total_error += error

    quantization_error = total_error / len(inputX)
    return quantization_error

#  I just named it as predict data, but it basically finds the confidence matrix with respect to the activation values
def predictData(weightMatrix, inputX, outputY, grid):
    confidenceMatrix = np.zeros((grid, grid))

    # each data unit is passed through euclidean distance and neighborhood function and confidence matrix is derived from here.
    for i, dataunit in enumerate(inputX):
        label = outputY[i]
        minIndex = bestMatchingNeuron(weightMatrix, dataunit, grid) # finds the minimum index associated with the input vector

        neighbourhood = gaussianFunctionWrap(minIndex, grid)  # neighborhood activation values

        if label == 0:
            confidenceMatrix += neighbourhood # summing up the nighbourhood influence values for each input vector.
        elif label == 1:
            confidenceMatrix -= neighbourhood # subtracting the neighbourhood influence values for each input vector.
    return confidenceMatrix

def plotErrorTrend(errors):
    iterations = [i * 100 for i in range(len(errors))]  # Scale x-axis by 100
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, errors, marker='o', linestyle='-', color='b')
    plt.title("Quantization Error Over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Quantization Error")
    plt.grid(True)
    plt.show()

# Normalizes input data between 0 and 1.
def minMaxNormalization(inputX, featureRange=(0,1)):
    minVal, maxVal = featureRange

    minData = np.min(inputX, axis = 0) # finds minimum along first dimension
    maxData = np.max(inputX, axis = 0) # finds maximum along first dimension

    # Normalization Process
    scale = (maxData - minData) 

    scale[scale == 0] = 1

    normalizedData = (inputX - minData) / scale

    normalizedData = normalizedData * (maxVal - minVal) + minVal

    return normalizedData

# Heat map visualization
def visualizeHeatMap(confidenceMatrix, gridSize):

    plt.imshow(confidenceMatrix, cmap='bwr_r', origin="upper", vmin=0, vmax=1)
    plt.colorbar(label="Confidence Level")
    plt.title("SOM Heat Map")
    plt.xlabel(f"Grid {gridSize}")
    plt.ylabel(f"Grid {gridSize}")
    plt.show()

# Main Function - SOM Implementation
if __name__ == "__main__":
    config = loadConfig()

    # Grid Size - fetches from config.conf file
    gridSize = int(config['SOM']['grid'])
    print(f"Grid: {gridSize} X {gridSize}")

    # fetches the datafile from config
    dataFile = config['SOM']['data_file']
    print("Dataset: ", dataFile)

    # Fetches Learning rate from config
    learningRate = float(config['SOM']['learningRate'])
    print("Learning Rate: ", learningRate)

    # Fetches Max iterations from config
    maxIterations = int(config['SOM']['maxIterations'])
    print("Max Iterations: ", maxIterations)

    # @ sampleSize - how many data samples
    # @ binSize - input dimensions, how many features are there
    # dataStr - x and y data rows
    sampleSize, binSize, dataStr = loadDataFile(dataFile)

    # splits dataset and converts to a numpy array
    inputX, outputY = numpyConversion(dataStr, binSize)
    
    # initializing weights
    weightMatrix = initializeWeights(gridSize, binSize)

    # Normalizes input between 0 and 1
    inputX = minMaxNormalization(inputX, featureRange=(0,1))

    # Training SOM and quantization Error
    weightMatrix, errors = trainingSOM(weightMatrix, inputX, gridSize, learningRate=learningRate, maxIterations=2000)

    # Confidence Matrix calculated from activation 
    confidenceMatrix = predictData(weightMatrix, inputX, outputY, gridSize)
    print(confidenceMatrix) 
    # Normalizes confidence matrix within 0 to 1.
    normalizedMatrix = normalizedConfidence(confidenceMatrix)
    print(normalizedMatrix)

    # Heat map using matplot library
    visualizeHeatMap(normalizedMatrix, gridSize)

    # This visualizes the quantization error.
    plotErrorTrend(errors)

