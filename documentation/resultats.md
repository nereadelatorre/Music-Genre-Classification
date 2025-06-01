# Resultats de l'Entrenament

En aquest document expliquem el progrés extens de les millores aplicades progressivament, dels resultats obtinguts per les diferents arquitectures probades i les conclusions.

## Millores aplicades progressivament
### CNN(1D) + GRU
En aquesta primera secció expliquem els resultats de l'arquitectura CNN(1D) + GRU. Primerament es va entrenar l'arquitectura per comprovar el seu rendiment. Posteriorment començar amb una cerca d'hiperparàmetres.

Amb l’objectiu de millorar el rendiment del model de classificació de gèneres musicals, s’ha realitzat una cerca mitjançant una **cerca bayesiana** d’hiperparàmetres (permet explorar l’espai de configuracions de manera més eficient que la cerca aleatòria) utilitzant **Weights & Biases**.

La configuració utilitzada per a aquesta cerca s’ha definit al fitxer `sweep.yaml`, amb l’objectiu de **minimitzar la pèrdua de validació (`val_loss`)**. A continuació es mostren els paràmetres i els seus valors explorats:

```yaml
# sweep.yaml
program: main.py

method: bayes
metric:
  name: val_loss
  goal: minimize

parameters:
  epochs:
    value: 30

  batch_size:
    values: [16, 32, 64, 128]

  learning_rate:
    values: [1e-4, 3e-4, 5e-4, 7e-4, 1e-3]

  hidden_dim:
    values: [64, 128, 256]

  dropout:
    values: [0.2, 0.3, 0.4, 0.5]

  num_layers: #en el RNN
    values: [1, 2]
```

Un cop obtenim la millor combinació, executem el codi durant 30 epochs. Hem recollit diferents mètriques, com la pèrdua mitjana del train (epoch_avg_loss), la pèrdua de validació (val_loss) i la pèrdua per batch (batch_loss).

<img src="\imatges\graphics\CNN1D-GRU1.jpg" alt="Gràfic pèrdues" width="700"/>


Veiem que la pèrdua de validació comença a augmentar mentre la d'entrenament continua baixant. Això és un símptoma d’**overfitting**.

***


Davant l’augment progressiu de la pèrdua de validació observat en l’execució anterior, plantegem una nova execució aplicant **tècniques de regularització** per millorar la capacitat de generalització del model. L’objectiu és reduir la diferència entre la pèrdua d’entrenament i la de validació.

En aquesta nova execució del model, s’han incorporat diverses millores per tal d’afavorir una millor generalització i estabilitat durant l’entrenament. Concretament, s’ha fixat el nombre d’**èpoques en 30**, amb un mecanisme d’**early stopping** de paciencia 6 per evitar un entrenament innecessàriament llarg si no hi ha millores. La learning rate inicial s’ha establert a 1e-3, amb un **weight decay de 1e-4** aplicat a l’**optimitzador Adam** com a tècnica de **regularització L2**. També s’ha afegit un scheduler de tipus **ReduceLROnPlateau**, per reduir dinàmicament la taxa d’aprenentatge quan la pèrdua de validació es manté estable. A nivell d’arquitectura, el model CNN-GRU utilitza ara una **dimensió oculta de 128 neurones** i un **dropout del 40%**, per ajudar a reduir el sobreajustament.

Tornem a entrenar el model per veure les millores:

<img src="\imatges\graphics\CNN1D-GRU2.jpg" alt="Gràfic pèrdues" width="500"/>

En el primer gràfic, observem una millora clara respecte a l’execució anterior. En el segon gràfic, que mostra l’evolució del learning rate, podem veure com aquest es redueix en dues fases ben diferenciades.

També observem que la mètrica de val_accuracy mostra una tendència ascendent clara i sostinguda, arribant a superar el **65%**.


***

En la nova execució a continuació, hem fet dues modificacions addicionals amb l’objectiu de seguir millorant la capacitat del model per generalitzar.

D’una banda, hem aplicat **data augmentation** sobre els espectrograms per incrementar la variabilitat de les mostres d’entrenament. Aquesta tècnica permet exposar el model a versions una mica alterades de les mateixes dades, ajudant-lo així a no sobreajustar-se i a ser més robust.

D’altra banda, hem realitzat una **selecció dels gèneres** musicals presents al dataset. Eliminem aquells tracks que pertanyen als gèneres amb menys de 200 mostres (tenint en compte que la més freqüent en té 14182), concretament: *blues*, *easy listening*, *electronic* i *soul-RnB*. Aquesta reducció ens permet centrar l’entrenament en classes amb representació una mica més equilibrada, millorant potencialment el rendiment del model.

Els resultats han estat:

<img src="\imatges\graphics\CNN1D-GRU3.jpg" alt="img" width="500"/> 


Després d’aplicar tècniques de regularització i mitigació de l’overfitting, veiem que la corba de la loss s’ha estabilitzat notablement: tant la train loss com la validation loss disminueixen de manera coordinada, evitant l’increment progressiu que es produïa la primera execució.

A més, concretament en aquest últim resultat, obtenim una milloria en la loss i l'accuracy del validation, augmentant a un **69%**.

***


### CNN(1D) + LSTM

Després d’haver obtingut bons resultats amb l’arquitectura basada en CNN(1D) + GRU, vam voler provar una nova arquitectura de RNN. En concret, hem mantingut la mateixa estructura de convolucions 1D, però substituint la capa GRU per una **LSTM**, primerament **unidireccional**. 

Per fer el primer entrenament, vam partir dels millors hiperparàmetres trobats amb la GRU. La precisió de validació augmenta inicialment de forma pronunciada fins assolir valors propers al **68%**, el que ens indica que no hem millorat. 

***
Per capturar millor les dependències temporals, hem utilitzat una LSTM **bidireccional**, seguida de dues capes de global pooling —average i max—, abans de la classificació final (per captar la tendència global de l’activitat temporal en l’espectrograma). A més, hem ajustat el comportament de l’aprenentatge amb un **ReduceLROnPlateau més agressiu** per afavorir una adaptació més ràpida i una millor convergència quan la validation loss s’estanca.

Veiem que els canvis no han estat gaire significatius:

<img src="\imatges\graphics\CNN1D-LSTM2.jpg" alt="img" width="500"/> 

Veiem que en el learning rate, es produeixen dues baixades durant l'entrenament, les quals afavoreixen una reducció suau però més pronunciada de les pèrdues en els trams posteriors. En relació amb la accuracy de validació, observem un comportament molt similar al de l'execució anterior: un augment ràpid a l'inici seguit d'una estabilització al voltant del **68%**.

***

Veiem que a partir del pas 400k (època 10) la disminució de la loss és molt més lenta, per tal d'evitar això, provem  una **reducció del learning rate menys permissiva** (ajustant el paràmetre threshold a 1e-4 i establint threshold_mode='abs'). I considerem aplicar una estratègia alternativa com **StepLR** amb un pas de 10 èpoques, que redueix el learning rate de forma més sistemàtica i controlada.

<img src="\imatges\graphics\CNN1D-LSTM3.jpg" alt="img" width="500"/> 

Amb aquestes millores, aconseguim una reducció de la loss important i un accuracy al validation del **69%**, indicant una petita millora en la capacitat de generalització del model.

***

Davant dels bons resultats obtinguts amb l’ús del StepLR, decidim fer un **bayesian search** per 
optimitzar encara més el rendiment d’aquesta xarxa i explorar el seu potencial. 
Hem variat els mateixos hiperparàmetres que en la primera cerca, afegint els que involucren
el StepLR i el ReduceOnPlateau:
```yaml
  reduce_lr_factor: #factor que redueix el LR en el ReduceOnPlateau
    values: [0.5, 0.7]
  step_lr_step_size: #número de steps en StepLR
    values: [5, 10]
  step_lr_gamma: #factor que redueix el LR en el StepLR
    values: [0.5, 0.7]
```
Un cop hem fet l'execució del model amb els valors dels millors hiperparàmetres, obtenim que la validation accuracy assoleix un valor molt destacat de **71,3%**, el qual és la més alta que hem obtingut fins ara, cosa que reafirma el bon comportament de la regularització.

***
Com a primera conclusió d'aquests primers models amb CNN de 1 dimensió, veiem que aconseguim pràctiment el **mateix rendiment** tant per la RNN GRU com amb la LSTM. Aquí es pot veure la loss i l'accuracy del validation d'ambdues arquitectures:

<img src="\imatges\graphics\Comparacio1D.jpg" alt="img" width="600"/> 

***
En aquesta secció següent fem un canvi d'arquitectura en la CNN. Ara provem amb una CNN de 2 dimensions, que, com bé expliquem a arquitectures.md, esperem millors resultats ja que podrem reconèixer sons amb variacions freqüencials (no només temporals).

### CNN(2D) + LSTM


Comencem avaluant aquesta nova arquitectura amb la RNN de LSTM. En un inici vam provar amb una simplificació de l'estructura que amb 1D, la mateixa con
configuració de hiperparàmetres, etc. Vam obtenir:

<img src="\imatges\graphics\CNN2D-LSTM.jpg" alt="img" width="600"/> 

No veiem un comportament molt diferenciat, però si observem un valor en l'accuracy del **71%**. El que ens indica que ja obtenim bon resultat de partida, que probablment podrem millorar amb la cerca d'hiperparàmetres.

Abans però de fer la cerca, vam provar de fer petits canvis en aquesta arquitectura,
com afegir una MaxPool després de les ReLu en les convolucions i fer un Dropout a la LSTM.

Aquests canvis no van implicar cap canvi significatiu en el comportament del model, però si vam aconseguir reduïr l'overfitting, aconseguint un model més estable i fiable. Abans d'iniciar el hyperparameters search, provem amb la RNN GRU per veure si trobem un canvi en el rendiment.

### CNN(2D) + GRU
Amb aquesta nova arquitectura obtenim uns molts bons resultats de base, **72%** d'accuracy:

<img src="\imatges\graphics\CNN2D-GRU.jpg" alt="img" width="600"/>

Tot i que veiem certa tendència a overfitting en la corba de loss, això ho vam solucionar de la mateixa forma que amb la LSTM, (MaxPool després de les ReLu en les convolucions i fer un Dropout a la GRU) permetent-nos tant reduir la complexitat com reforçar la generalització del model:

<img src="\imatges\graphics\CNN2D-GRU2.jpg" alt="img" width="300"/>

De manera que, finalment, hem realitzat la mateixa cerca de paràmetres amb ambdues arquitectures. Hem utilitzat els mateixos paràmetres i rangs que en l'anterior cerca. 

***

## Altres intents "millores" i canvis no implementats

En avaluar el model, vam detectar que uns certs gèneres musicals, els que tenien menys mostres en el dataset, obtenien un accuracy bastant pitjor en comparació amb les classes majoritàries. Degut al desbalanceig en la distribució de les dades, trobem més dificultat en la classificació correcta de les categories amb menys exemples.

Per tractar aquest problema, vam ajustar la funció de pèrdua (**CrossEntropy ponderada**), assignant pesos inversament proporcionals al nombre de mostres per classe. D'aquesta manera, el model prioritzaria aprendre dels gèneres infrarepresentats durant l'entrenament.

Si bé la tècnica va millorar el rendiment en les classes minoritàries, va tenir un efecte advers en les majoritàries: la seva loss va augmentar molt, la qual cosa va acabar perjudicant el rendiment global del model:

<img src="\imatges\graphics\crossEntropy.jpg" alt="img" width="500"/>

Veiem la comparació entre el model previ (rosa) i el model amb el Cross Entropy ponderat aplicat (marró).
Vam concloure que el balanceig mitjançant pesos en la funció de pèrdua, encara que és útil per a equilibrar mètriques per classe, no necessàriament optimitza el resultat general quan el desbalanceig és molt accentuat.

***

També vam voler probar amb l'arquitectura individiual de la CNN en 2 dimensions, per veure si hi havia molta diferència amb aplicar o no les xarxes recurrents.

En aquest gràfic mostrem la comparació de la pèrdua de la CNN(2D) (color verd), i CNN(2D) amb GRU (color groc):

<img src="\imatges\graphics\CNN(2D).jpg" alt="img" width="500"/>

El model CNN+GRU mostra un rendiment superior respecte a la CNN sola. La CNN captura patrons espectrals locals en els espectrograms, però la incorporació de la GRU permet modelar la dinàmica temporal d’aquests patrons. La GRU, mitjançant el seu mecanisme de portes (reset i update), filtra informació seqüencial rellevant, elimina soroll i prioritza transicions rítmiques i harmòniques significatives. Això millora la capacitat predictiva del model sense augmentar significativament el cost computacional, resultant en una menor pèrdua i una estabilitat més gran.

***

## Anàlisi final dels resultats

Per tancar l’estudi, fem una última execució de les 4 arquitectures (amb els 
millors hiperparàmetres i la millor estructura trobada) durant 40 èpoques.

Per mesurar el rendiment dels nostres models no només volem utilitzar l’accuracy, sinó utilitzar **diverses mètriques** perquè
cada una aporta una perspectiva diferent del comportament del model. L’**accuracy** 
mesura la proporció global d’encerts, però ens pot amagar desequilibris en classes 
minoritàries. El **F1-macro** tracta totes les classes per igual i ens indica com de 
bé equilibra precisió i recall en els gèneres menys freqüents, mentre que el 
F1-weighted pondera cada classe segons la seva representació, reflectint millor 
l’impacte global dels errors en gèneres molt i poc comuns. Així, combinant-les 
obtenim una visió més completa i justa de la capacitat real de generalització del 
model:

$$
\mathrm{Accuracy} = 
\frac{\sum_{c=1}^C \mathrm{TP}_c + \mathrm{TN}_c}
     {\sum_{c=1}^C (\mathrm{TP}_c + \mathrm{TN}_c + \mathrm{FP}_c + \mathrm{FN}_c)}
\times 100\%
$$

Fórmula del F1 per a cada classe \(c\):
$$
\mathrm{F1}_c = 2 \times \frac{\mathrm{Precision}_c \;\times\; \mathrm{Recall}_c}{\mathrm{Precision}_c + \mathrm{Recall}_c}
$$

On per a cada classe \(c\):
$$
\mathrm{Precision}_c = \frac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FP}_c}, 
\quad
\mathrm{Recall}_c = \frac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FN}_c}
$$

A partir d’aquí obtenim les dues versions que utilitzem:

- **F1-macro** (pes igual per a cada classe):
  $$
  \mathrm{F1\mbox{-}macro} = \frac{1}{C} \sum_{c=1}^C \mathrm{F1}_c
  $$

- **F1-weighted** (pondera cada classe segons la seva freqüència \(n_c\)):
  $$
  \mathrm{F1\mbox{-}weighted} = \sum_{c=1}^C \frac{n_c}{\sum_{k=1}^C n_k} \;\mathrm{F1}_c
  $$

### Resultats i anàlisi quantitativa
Com que el comportament dels gràfics de totes les mètriques estudiades evoluciona igual respecte els models entrenats, a continuació mostrem les corbes de la la loss i del F1 macro obtingudes, ja que aquesta mètrica ens ofereix una visió clara de la capacitat del nostre model per reconèixer correctament cada gènere, especialment els menys representats.

<img src="\imatges\results\results.png" alt="img" width="700"/>

Veiem que totes quatre arquitectures segueixen pràcticament la **mateixa trajectòria**: en els primers 200 000 passos aprenen amb rapidesa (la validation loss baixa i el F1-macro puja amb força pendent) i a partir d’uns 600 000–800 000 passos comencen a estabilitzar-se al voltant d’un F1-macro d’un 0.60-0.62. També observem que hi ha petites oscil·lacions puntuals, però tots convergeixen a resultats molt similars. Això ens confirma que, després de l’etapa inicial d’aprenentatge ràpid, les millores addicionals venen més de polir hiperparàmetres o augmentacions que no pas de canviar d’arquitectura.


En aquesta taula mostrem els valors de totes les mètriques obtinguts amb el conjunt de test:

| Arquitectura     | Accuracy (%) | F1-weighted (%) | F1-macro (%) |
|------------------|-------------:|----------------:|-------------:|
| CNN 1D + GRU     |        69.10 |            67.4 |         55.6 |
| CNN 1D + LSTM    |        69.69 |           68.64 |         59.8 |
| CNN 2D + LSTM    |        70.83 |           69.70 |        60.77 |
| CNN 2D + GRU     |        71.40 |           70.28 |        61.90 |

Després d’entrenar quatre arquitectures diferents combinant CNNs (1D i 2D) amb LSTM i GRU, observem que totes obtenen resultats similars en les mètriques, amb una precisió de validació al voltant del **70%**. Tot i que el model basat en **CNN 2D amb GRU** presenta una lleugera avantatge, **les diferències no són significatives** entre els models.

Observem també diferències de valors entre les mètriques. Això es deu principalment al **desequilibri de classes**. En l’accuracy observem la influència dels gèneres predominants ben classificats (per exemple, Pop o Rock). L’F1-weighted corregeix parcialment aquest efecte ponderant cadascuna de les F1-score per la seva freqüència, però encara dona més pes a les classes majoritàries. En canvi, l’F1-macro tracta totes les classes per igual, mostrant així amb més claredat el rendiment en gèneres minoritaris. El valors més baixos d'aquesta mètrica, llavors, ens diuen que, en general, **les classes amb menys exemples obtenen pitjors resultats**.

Per analitzar més en profunditat aquestes confusions en la classificació, observem la **matriu de confusió** (mostrem la del model amb CNN(2D) i GRU):

<img src="\imatges\results\confusion_matrix.png" alt="img" width="600"/>

Veiem que les classes més ben classificades són **Rock**, **Hip-Hop**, **Old-Time / Historic** i **Classical**, amb valors elevats a la diagonal. En canvi, veiem una confusió clara en
molts gèneres (com Folk, Jazz i Instrumental) que es confonen amb Rock, probablement es degui a que Rock és el gènere més predominant (amb més de 14000 tracks), i per tant el model capta moltes més variacions de característiques, fàcils de confondre amb els gèneres no tan diferenciats
o amb patrons no tant recognoscibles. Un cas clar on passa lo contrari és amb **Old-Time/Historic**, que és el gènere millor classificat (degut a que és un gènere molt diferencait dels altres).

També veiem que hi ha confusions notables entre classes que comparteixen tècniques o instruments: Instrumental sovint es confon amb Experimental, Electronic o Pop. Experimental rep bona classificació però també es confón amb Electronic i  Pop. Finalment, en Spoken veiem un 0,30 de falsos positius cap a Experimental, probablement pels timbres no musicals o distorsionats que comparteixen. 

Aquesta anàlisi confirma que el model captura bé els patrons més clars però té dificultats i confusions per la similitud estructural entre alguns d’aquests gèneres.


### Anàlisi qualitatiu del comportament del model

Per dur a terme aquest anàlisi qualitatiu, ho hem dividit en dues parts: primer hem fet una **visualització de les representacions de les classes** (gèneres) amb **t-SNE**, i d'altra banda hem optat per **comparar visualment els mel-espectrograms** per observar les diferències visuals entre gèneres.

**1. Visualització de les representacions amb t-SNE:**

Per tal de comprendre com el nostre model separa els diferents gèneres en l’espai de característiques, vam extreure els embeddings del penúltim layer de la xarxa i els vam projectar a dues dimensions amb t-SNE. El codi carrega el dataset filtrat, obté les sortides del feature extractor i l’encoder (hem utilitzat GRU) per a cada espectrograma, i aplica t-SNE amb perplexitat 30 i 1 000 iteracions per obtenir una disposició 2D. A la figura, cada punt representa una pista i està pintat segons el seu gènere.

<img src="\imatges\results\tsne.png" alt="img" width="800"/>


En el resultat observem clarament quins són els gèneres ben definits gràcies a les agrupacions més compactes, com per exemple Rock, Hip-Hop i, sobre tot Old-Time / Historic. Les classes minoritàries (amb menys del 5 % del dataset), en general, apareixen disperses i, en alguns casos, s’hi ajunten amb gèneres veïns (com per exemple Pop que veiem que es troba distribuït en gairebé tot el pla), mostrant-nos que formen nuclis molt més difusos. 

Aquesta separació visual de les classes ens confirma que el nostre model ha après característiques discriminants per als gèneres principals i gèneres molt diferenciats (com hem mencionat a l'anàlisi quantitatiu), però que encara presenta dificultats per a aquells amb alta variabilitat interna o patrons no tan diferenciats.


**2. Visualització mel-spectrogramas:**

Per últim hem optat per comparar visualment els mel-espectrograms des de la mateixa perspectiva amb què una CNN els interpreta: *com a imatges*. Tot i que un espectrograma mel conté informació acústica específica, el model no "entén" que es tracta de senyals sonors, sinó que aprèn patrons visuals (textures, formes, contrastos) dins d’una imatge en escala de grisos. Per això, l’anàlisi comparativa visual dels espectrograms ens permet aproximar-nos a com el model pot confondre o reconèixer determinats gèneres musicals

*Identitat visual consistent:*

<img src="/imatges/mel_spectrograms/RockTracks.drawio.png" alt="Rock Tracks" width="800"/>

Tot i que diversos gèneres musicals poden tenir una identitat clara, com Hip-Hop, Classical o Rock, en aquest anàlisi ens centrarem en fer la comparació amb el cas del **Rock**, ja que presenta una de les taxes més altes de classificació correcta (88%), i és també un dels gèneres amb que més es confonen els altres. 

Sin ens fixem en els mel-espectrograms dels tracks de Rock, les tres imatges mostren una distribució de colors molt similar, amb una textura densa i tonalitats càlides repartides de manera uniforme, el que ens permet entendre des d’un punt de vista visual l'alta precisió de classificació d'aquest gènere. Aquesta homogeneïtat visual entre els exemples del mateix gènere fa que, en ser interpretats com imatges per la CNN, el model pugui reconèixer fàcilment patrons recurrents i classificar-los amb seguretat.

*Confusions per similitud visual:*

<img src="/imatges/mel_spectrograms/pop_tracks.drawio.png" alt="Pop Tracks" width="500"/>

En contraposició al cas del Rock, observem que el model tendeix a confondre sovint el gènere **Pop** amb Rock (fins a un 31% dels casos). Aquesta confusió es pot explicar des d’un punt de vista visual, ja que els mel-espectrogrames dels tracks de Pop amb el ID 1642 i 10 mostren una estructura visual molt semblant als del Rock. Aquestes tenen una distribució de colors gairebé idèntica i una textura igualment contínua i densa. En tractar els espectrogrames com imatges, la CNN no capta la intenció musical, sinó que respon a patrons visuals repetits. Això pot portar el model a associar incorrectament els exemples de Pop com si fossin Rock.

*Diferenciació clara amb altres gèneres:*

<img src="/imatges/mel_spectrograms/ElectronicTracks.drawio.png" alt="Electronic Tracks" width="500"/>

A diferència del cas de Pop, el gènere **Electronic** es confon molt poc amb Rock (només un 6%). Aquesta separació es pot entendre fàcilment des d’un punt de vista visual. En els espectrogrames mel dels tracks d'Electronic (com els ID 149701 i 1482), s’observen imatges més fosques en general, amb bandes visuals més segmentades i patrons repetitius molt marcats. Aquests espectres mostren una distribució de colors i formes molt diferent de la del Rock. Aquesta diferenciació visual clara facilita que la CNN identifiqui amb facilitat els exemples de Electronic i no els confongui amb altres gèneres.


En conclusió, l’anàlisi visual dels espectrogrames mel ens permet entendre millor les confusions del model. Veiem que confirmem, en part, les conclusions dels anàlisis prèvis: Aquells gèneres amb representacions espectrals consistents i distintives (com Rock o Electronic) són més fàcilment classificables, mentre que aquells amb trets compartits o difusos tendeixen a generar més errors.


## Conclusions

Les diferències en les mètriques entre les quatre arquitectures són petites (entre 69 % i 72 %). Tot i que les versions 2D tendeixen a quedar lleugerament per sobre, no hi ha un salt qualitatiu intuïtiu que permeti declarar-ne un clar guanyador.

Atès que el rendiment depèn fortament del dataset (nombre de mostres per gènere, qualitat d’enregistrament, etc.), no podem afirmar que un conjunt de paràmetres o una arquitectura sigui la “correcta” de manera absoluta. Cada model té el seu propi equilibri entre capacitat de representació i risc d’overfitting.
Hem treballat sobre FMA Large, que presenta un desbalanç de classes notable (gèneres amb milers de mostres vs. altres amb centenars). Aquesta distribució pot haver frenat la pujada d’accuracy per als models més senzills. És probable que, si haguéssim emprat FMA Small (més homogeni), les corbes i els valors finals fossin diferents, i potser els “millors” paràmetres ho serien en un altre context.

També hem confirmat que el model aprèn a distingir bé els gèneres amb patrons visuals molt consistents, mentre que té més dificultats amb els gèneres de representació espectral difusa o amb trets compartits.


En conclusió, tots quatre models ofereixen un rendiment sòlid i molt proper entre ells. La decisió d’usar un o altre dependrà de factors addicionals com ara temps d’entrenament, capacitat de càlcul disponible i l’objectiu final (per exemple, prioritzar lleugeresa per a dispositius amb recursos limitats).