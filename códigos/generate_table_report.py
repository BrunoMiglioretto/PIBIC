import pandas as pd

datasets = {
    "Atrial Fibrillation": "AtrialFibrillation_results.csv",
    "Cricket": "Cricket_results.csv",
    "ERing": "ERing_results.csv",
    "Face Detection": "FaceDetection_results.csv",
    "UWave Gesture Library": "UWaveGestureLibrary_results.csv"
}

representations = [
    "CWT", 
    "MTF", 
    "GASF", 
    "GADF", 
    "RP", 
    "FIRTS",
]

operations = [
    "sum", 
    "subtraction", 
    "dot_product", 
    "element_wise",
]

# espectrograma

convolutions = ["", "MiniRocket",]

results = []


for rep in representations:
    print(f"{rep} - Ridge")
    for ds_name, file_name in datasets.items():
        df = pd.read_csv(file_name)  
        r = df[(df["convolution_algorithm"].isnull()) & (df["representation"] == rep)]

        acc = r[["accuracy"]].T
        print(acc.to_string(header=False, index=False))

        results.append(r)

    print("-----------------------------------------------------")

for rep in representations:
    print(f"{rep} - MiniRocket")
    for ds_name, file_name in datasets.items():
        df = pd.read_csv(file_name)  
        r = df[(df["convolution_algorithm"] == "MiniRocket") & (df["representation"] == rep)]

        acc = r[["accuracy"]].T
        print(acc.to_string(header=False, index=False))

        results.append(r)

    print("-----------------------------------------------------")

# for result in results:
#     print("------------------")

#     print(result[["dataset", "representation", "operation", "convolution_algorithm", "accuracy"]])
