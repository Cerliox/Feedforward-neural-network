# CMakeList.txt: CMake-Projekt für "GeneticNeuralNetworks". Schließen Sie die Quelle ein, und definieren Sie
# projektspezifische Logik hier.
#
cmake_minimum_required (VERSION 3.8)

# Fügen Sie der ausführbaren Datei dieses Projekts eine Quelle hinzu.
add_executable ("NeuralNetworks" "NeuralNetworks.cpp" "NeuralNetworks.h" "Clock.h")

# TODO: Fügen Sie bei Bedarf Tests hinzu, und installieren Sie Ziele.
