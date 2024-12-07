# Detectia stresului din imagini

## Tabel de analiza

| Nr. | Autor/An | Titlu articol/proiect | Domeniu | Tehnologii Utilizate | Metodologie | Rezultate         | Limitari | Comentarii suplimentare |
| -- | ------------------------ | --------------- | ---------- | ------------------ | ------------- | ------------------ | ---------- | ------------------------ |
| 1 | 2023, Jaramillo-Quintar D., Gomez-Reyes J, Morales-Hernandez L., Dominguez-Trejo B., Rodriguez-Medina D., Cruz-Albarran I. | Automatic Segmentation of Facial Regions of Interest and Stress Detection using Machine Learning | Biomedical Engineering | TSST, HOG, SVM, Python 3, Thermography | automatic ROI selector for thermal face images, the process of obtaining average temperature values within the ROIs, and the intelligent classifier for baseline, stress, and relaxation![image](https://github.com/user-attachments/assets/a16a0d18-2f6d-4060-8fd7-1851ec3f1ecb) | ![image](https://github.com/user-attachments/assets/1ba96cd3-4446-41e9-9a95-a86d263d5ec7) ![image](https://github.com/user-attachments/assets/81bd7662-9a13-4fef-a9b4-bff9ba84bb07) | a blurred, non-frontal or obstructed image limits the correct selection of ROIs and the acquisition of the mean temperature | |
| 2 | 2021, Katarzyna Baran | Stress detection and monitoring basd on low-cost mobile thermography | Mental Health | OpenCV, thermal imaging camera, SVN, HSI, Flir VividIR image processing, ART Hankfit S-FIT18 | ![image](https://github.com/user-attachments/assets/8b6492ca-49c5-4b3c-b049-cf5e00bb5720)  feature cascade with Haar classifier, model 68 face landmark detection, HOG feature detection, face point detection with shape predictor, ROI drawing with frame comparison, real-time detection of one or more colors | ![image](https://github.com/user-attachments/assets/70036c0a-7b9d-4d68-adb9-c0eb132b2c18) ![image](https://github.com/user-attachments/assets/0c66e9e1-958c-4510-9cf4-980f36a41049) S/C/E means: S-start, C-center, E-end and refers to the culmination of stress, where there is a sudden and temporary change in the face temperature value | most models incorrectly recognized the thermal features of the face or did not indicate areas of changes. The most appropriate method for the pilot study turned out to be the method of detecting areas of the dominant color, with a variable range – in this case, the red color | |
| 3 |
| 4 |
| 5 |

## Schema bloc
![image](https://github.com/user-attachments/assets/8f3f37b8-a8e0-4a94-afc0-48c46bb7eea8)



