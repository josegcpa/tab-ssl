import os


def save_csv(model, name, idx):

    try:
        os.makedirs('results')
    except:
        print('Results folder already exists.')

    outputFilename = os.path.join('results', "StdGP_" + name + "_" + str(idx) + ".csv")
    if not os.path.exists(outputFilename):

        accuracy = model.getAccuracyOverTime()
        waf = model.getWaFOverTime()
        kappa = model.getKappaOverTime()
        mse = model.getMSEOverTime()
        size = model.getSizeOverTime()
        model_str = str(model.getBestIndividual())
        times = model.getGenerationTimes()

        results = []

        results.append((accuracy[0],
            accuracy[1],
            waf[0],
            waf[1],
            kappa[0],
            kappa[1],
            mse[0],
            mse[1],
            size,
            times,
            model_str
        ))


        # Write output header
        file = open(outputFilename, "w")
        file.write("Attribute,Run,")
        for i in range(model.max_generation):
            file.write(str(i) + ",")
        file.write("\n")

        attributes = ["Training-Accuracy", "Test-Accuracy",
                      "Training-WaF", "Test-WaF",
                      "Training-Kappa", "Test-Kappa",
                      "Training-MSE", "Test-MSE",
                      "Size",
                      "Time",
                      "Final_Model"]

        # Write attributes with value over time
        for ai in range(len(attributes) - 1):
            for i in range(len(results)):
                file.write("\n" + attributes[ai] + "," + str(i) + ",")
                file.write(",".join([str(val) for val in results[i][ai]]))
            file.write("\n")

        # Write the final models
        for i in range(len(results)):
            file.write("\n" + attributes[-1] + "," + str(i) + ",")
            file.write(results[i][-1])
        file.write("\n")

        # Write some parameters
        file.write("\n\nParameters")
        file.write("\nOperators," + str(model.operators))
        file.write("\nMax Initial Depth," + str(model.max_initial_depth))
        file.write("\nPopulation Size," + str(model.population_size))
        file.write("\nMax Generation," + str(model.max_generation))
        file.write("\nTournament Size," + str(model.tournament_size))
        file.write("\nElitism Size," + str(model.elitism_size))
        file.write("\nDepth Limit," + str(model.max_depth))
        file.write("\nWrapped Model," + model.model_name)
        file.write("\nFitness Type," + model.fitnessType)
        file.write("\nThreads," + str(model.threads))
        #file.write("\nRandom State," + str(list(range(runs))))
        file.write("\nDataset," + name)

        file.close()
    else:
        print("Filename: " + outputFilename + " already exists.")
