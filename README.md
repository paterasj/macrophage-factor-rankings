# Inferring factor importance in explainable AI model for macrophage immune cell polarization
COPASI model and accompanying LSTM code for downstream analysis accompanying the research article: Inferring factor importance in explainable AI model for macrophage immune cell polarization.

macrophage_simulation_model.cps is the COPASI model file used to generate widescale parameter scans for training the surrogate model. It can be loaded directly into COPASI and simulations can be performed.

lstm_surrogate_model.py is the Python script written to work directly in a Google Colab environment. With parameter scans from COPASI as input, this model trains a surrogate LSTM and outputs relative feature rankings to infer importance in influencing macrophage state transitions. The figures and factors displayed are written to mimic the presentation in the research article.
