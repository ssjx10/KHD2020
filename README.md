# KHD2020
Korea Health Datathon 2020
https://github.com/Korea-Health-Datathon/KHD2020

main.py
- 5 Fold CV

dummy_main.py
- split validation or full train

ensemble_main.py
- inference in ensemble model

ensemble2_main.py
- train a student models
- ex) DML

customEval.py
- Accuracy, Specificity, Sensitivity, Precision, Negative Predictable value, F1 score

nsml command
- nsml run -d Breast_Pathology -e main.py --shm-size 16G
- nsml model ls -j KHD005/Breast_Pathology/72
- nsml submit KHD005/Breast_Pathology/72 6
