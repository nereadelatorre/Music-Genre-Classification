## Funció test
***

### Objectius:
- Avaluar el model sobre el conjunt de test.
- Calcular mètriques de classificació: accuracy, F1-score macro, F1-score weighted.
- Generar i visualitzar la matriu de confusió.
- Exportar el model entrenat a ONNX.
- Enregistrar els resultats a wandb per al seu seguiment.
---

### Paràmetres:
| **Paràmetre** | **Descripció** |
|---------------|----------------|
| `model`       | Model PyTorch prèviament entrenat. |
| `test_loader` | DataLoader amb les dades de test. |
| `device`      | Dispositiu d’execució (`"cpu"` o `"cuda"`). |
| `save`        | Si és `True`, exporta el model en format ONNX. |

---

### Funcionament de la funció:

1. **Mode d'avaluació (`eval`)**  
   El model es posa en mode d’avaluació per assegurar que el model faci prediccions estables i coherents, desactivant parts que funcionen de manera aleatòria durant l'entrenament. A més, es desactiva el càlcul de gradients amb `torch.no_grad()` per reduir el consum de memòria i millorar l’eficiència durant la inferència.

2. **Inferència**  
   Es recorre el conjunt de test (`test_loader`) i es passen els espectrogrames pel model. 

3. **Recollida de resultats**  
   Es guarden totes les prediccions i etiquetes reals per calcular posteriorment mètriques.

4. **Càlcul de mètriques globals**  
   - **Accuracy**
   - **F1-score (weighted)**
   - **F1-macro**

5. **Matriu de confusió**  
   Es genera una matriu de confusió per analitzar quines classes es confonen més. Es visualitza en dos formats:
   - Comptes absoluts.
   - Matriu normalitzada per files (recall per classe).

   Es desa com a imatge `.png` i s’enregistra a Weights & Biases.

6. **Exportació a ONNX (opcional)**  
   Si es passa el paràmetre `save=True`, el model es converteix al format `.onnx`, que permet utilitzar-lo en altres entorns com TensorRT, ONNX Runtime o altres aplicacions.

7. **Registre a WandB**
   Les mètriques i la imatge de la matriu de confusió s’enregistren automàticament a wandb, per poder fer seguiment i comparació d'experiments.

---
### Mètriques d'avaluació:
Per tal d’avaluar el rendiment del model, hem utilitzat les següents mètriques:
- **Accuracy**: calcula la proporció total de prediccions correctes. Però pot resultar poc informativa en casos de desbalanceig entre classes (com és el nostre cas), ja que pot estar dominada per les classes majoritàries.
- **F1-score macro**: calcula el F1-score per a cada classe per separat i en fa la mitjana no ponderada. D’aquesta manera, tracta totes les classes per igual, independentment de la seva freqüència, i és especialment útil per detectar si el model funciona malament amb les classes minoritàries.
- **F1-score weighted**: Similar al F1 macro, però pondera el F1 de cada classe segons la seva proporció al conjunt de dades. Això ofereix una mesura més equilibrada tenint en compte el desbalanceig.

Amb aquestes mètriques es pot analitzar no només el rendiment global del model, sinó també si aquest és capaç de reconèixer bé les classes minoritàries.
