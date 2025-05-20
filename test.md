## Funció test:
***

### Objectius:
- Avaluar l'accuracy del model sobre dades no vistes.
- Calcular mètriques de classificació detallades (accuracy, precisió, recall, F1-score).
- Generar i visualitzar la matriu de confusió.
- Exportar el model entrenat a ONNX (opcional).
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
### Mode d’avaluació: 

- El model es posa en mode eval() per desactivar dropout i batch norm dinàmic. 
- No es calculen gradients (`torch.no_grad()`), ja que no s’està entrenant el model.
---

### Predicció: 

1. Es recorre el conjunt de test amb el `test_loader`.
2. Es passen els espectrograms pel model per fer inferència.
3. Es calcula la classe predita mitjançant `argmax`.
4. Es recullen totes les prediccions i etiquetes reals.
5. Es calcula l'**accuracy** total i s’imprimeix.
6. Es registra a wandb amb:
   ```python
    wandb.log({"test_accuracy": accuracy}) 
   ```
7. Si `save=True`, també es guarda a:

    ```python
    wandb.run.summary["test_accuracy"] = accuracy
    ```

---

### Mètriques de classificació:

- Es calcula el **classification report** amb `sklearn.metrics.classification_report`:
  - Inclou: precisió, recall, F1 per classe i macro/micro mitjanes.
  - Els gèneres musicals s’utilitzen com a noms de classe.
- Es mostra el report per consola.
- Es registra cada mètrica a `wandb.run.summary`.

---
### Accuracy per classe:

- Es calcula l’accuracy per classe mitjançant:

    ```python
    np.mean(np.array(preds)[mask] == i)
    ```

- Es mostra a consola i es registra a wandb amb:

    ```python
    wandb.log({"per_class_accuracy": per_class_acc})
    ```

---

### Matriu de confusió:

- Es genera la matriu amb `confusion_matrix`.
- Els noms de gèneres es mostren a l’eix X/Y.
- Es visualitza amb `seaborn.heatmap`.
- Es desa com a imatge `.png` i es registra a wandb:

    ```python
    wandb.log({"confusion_matrix": wandb.Image('confusion_matrix.png')})
    ```

---

### Exportació ONNX (opcional):

- Si `save=True`, el model es converteix a format `.onnx` amb `torch.onnx.export`.
- Permet eixos dinàmics (batch size, time steps).
- Es registra a wandb amb:

    ```python
    wandb.save("model.onnx")
    ```

- Si hi ha un error en l’exportació, es captura i s’imprimeix sense interrompre el test.

---


