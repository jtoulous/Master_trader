#import pandas as pd
#import copy
#import matplotlib.pyplot as plt
#import os

#from statistics import mean, stdev
from colorama import Fore, Style

def printLog(log):
    print(f'{Fore.GREEN}{log}{Style.RESET_ALL}')

def printError(error):
    print(f'{Fore.RED}{error}{Style.RESET_ALL}')

def printInfo(info):
    print(f'{Fore.BLUE}{info}{Style.RESET_ALL}')


#def get_labels(*dataframes):
#    full_df = pd.concat(dataframes, ignore_index=True)
#    labelsEncountered = []
#    nan_count = 0
#
#    for idx, row in full_df.iterrows():
#        if pd.isna(row['LABEL']):
#                nan_count += 1
#        if row['LABEL'] not in labelsEncountered:
#            labelsEncountered.append(row['LABEL'])
#    return labelsEncountered
#
#def saveConfig(network, data):
#    if os.path.exists("network.txt"):
#        os.remove("network.txt")
#    with open('network.txt', 'w') as network_file:
#        
#        network_file.write('Features Used:')
#        for feature in data.features:
#            network_file.write(f'{feature}')
#            if feature != data.features[-1]:
#                network_file.write(',')
#            else:
#                network_file.write('\n')
#
#        network_file.write('Architecture:')
#        for layer in network.layers:
#            network_file.write(f'{layer.shape}|{layer.activation}|{layer.weights_initializer}')
#            if layer != network.layers[-1]:
#                network_file.write(',')
#            else:
#                network_file.write('\n')
#
#        for l, layer in enumerate(network.layers):
#            network_file.write(f'Layer {l} {layer.type}:\n')
#            for n, neuron in enumerate(layer.neurons):
#                if neuron.label == None:    
#                    network_file.write(f'{n}:')
#                else:
#                    network_file.write(f'{neuron.label}:')
#                
#                for k, key in enumerate(neuron.weights.keys()):
#                    network_file.write(f'{key}={neuron.weights[key]}')
#                    if k != len(neuron.weights.keys()) - 1:
#                        network_file.write(',')
#                network_file.write('|')
#                network_file.write(f'{neuron.bias}\n')
#        
#        network_file.write('Means:')
#        for k, key in enumerate(data.normData['means'].keys()):
#            network_file.write(f'{key}={data.normData["means"][key]}')
#            if k != len(data.normData['means'].keys()) - 1:
#                network_file.write(',')
#            else:
#                network_file.write('\n')
#
#        network_file.write('Stds:')
#        for k, key in enumerate(data.normData['stds'].keys()):
#            network_file.write(f'{key}={data.normData["stds"][key]}')
#            if k != len(data.normData['stds'].keys()) - 1:
#                network_file.write(',')
#
#
#def printGraphs(meanCostHistory, precisionHistory):
#    meanCostTraining = meanCostHistory['train data']
#    meanCostValidation = meanCostHistory['valid data']
#    precisionTraining = precisionHistory['train data']
#    precisionValidation = precisionHistory['valid data']
#    epochs = [i for i in range(len(meanCostTraining))]
#
#    plt.figure(1) 
#    plt.plot(epochs, meanCostTraining, label='training loss')
#    plt.plot(epochs, meanCostValidation, label='validation loss')
#    plt.title("Loss")
#    plt.xlabel("Epochs")
#    plt.ylabel("Losses")
#    plt.legend()
#
#    plt.figure(2)
#    plt.plot(epochs, precisionTraining, label='training precision')
#    plt.plot(epochs, precisionValidation, label='validation precision')
#    plt.title("Precision")
#    plt.xlabel("Epochs")
#    plt.ylabel("Precision")
#    plt.ylim([0.5, 1])
#    plt.legend()
#
#    plt.show(block=False)
#
#    while plt.get_fignums():
#        plt.pause(0.5)
#
#
#def printEpochResult(epoch, total_epochs, meanCostTrain, meanCostValid):
#    print(f'epoch {epoch + 1}/{total_epochs} - loss: {meanCostTrain} - val_loss: {meanCostValid}')