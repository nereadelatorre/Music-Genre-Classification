## Resultats de l'Entrenament i la Cerca d'Hiperparàmetres

### Cerca d’Hiperparàmetres (Grid Search)

Amb l’objectiu de millorar el rendiment del model de classificació de gèneres musicals, s’ha realitzat una cerca aleatòria d’hiperparàmetres utilitzant **Weights & Biases (wandb)**.

La configuració utilitzada per a aquesta cerca s’ha definit al fitxer `sweep.yaml`, amb l’objectiu de **maximitzar la precisió de validació (`val_accuracy`)**. A continuació es mostren els paràmetres i els seus valors explorats:

```yaml
# sweep.yaml
program: main.py

method: random
metric:
  name: val_accuracy
  goal: maximize

parameters:
  epochs:
    value: 20

  batch_size:
    values: [16, 32, 64]

  learning_rate:
    values: [1e-4, 3e-4, 5e-4, 7e-4, 1e-3]

  hidden_dim:
    values: [64, 128, 256]

  dropout:
    values: [0.2, 0.3, 0.5]

  num_layers:
    values: [1, 2]
```
Aquest procediment ha permès avaluar múltiples combinacions d’hiperparàmetres de manera aleatòria. L’objectiu principal ha estat trobar una configuració que maximitzi el rendiment en validació i test, evitant l'*overfitting*.

### Resultats Obtinguts

Després de realitzar múltiples execucions (*runs*) amb diferents combinacions d’hiperparàmetres, la configuració amb millor rendiment ha estat:

- **Model**: CNN1D-GRU  
- **batch_size**: 16  
- **dropout**: 0.3   
- **hidden_dim**: 128  
- **learning_rate**: 0.0001  
- **num_layers**: 2  
- **Dataset**: FMA_large (`./fma_mel_features.h5`)  


Aquest *run* ha assolit la millor precisió en el conjunt de validació (`val_accuracy`) i també ha obtingut bons resultats en el conjunt de test, complint l’objectiu de generalització del model.
***
Un cop tenim els resultats del grid search, executem el codi durant 30 epochs amb els valors òptims trobats per als hiperparàmetres. Hem recollit diferents mètriques, com la pèrdua mitjana del train (epoch_avg_loss), la pèrdua de validació (val_loss) i la pèrdua per batch (batch_loss). A continuació, comentarem els gràfics resultants per analitzar el comportament del model durant l'entrenament i la validació.

<img src="imatges\img_2.png" alt="Gràfic pèrdues" width="700"/>


En el primer gràfic obtenim la comparació de les línies corresponents a la pèrdua d'entrenament (epoch_avg_loss) i la pèrdua de validació (val_loss) al llarg de les èpoques. S'observa que la pèrdua d'entrenament disminueix de manera consistent, cosa que indica que el model s'està ajustant bé als dades d'entrenament. No obstant això, a partir d’un cert punt, la pèrdua de validació comença a augmentar mentre la d'entrenament continua baixant. Això és un símptoma d’overfitting, ja que el model aprèn massa bé el conjunt d'entrenament però perd capacitat de generalització sobre les dades noves.

En el segon gràfic es mostra l’evolució de la pèrdua per batch (batch_loss) al llarg de l'entrenament. Podem veure que hi ha molta variabilitat entre batches, però en general hi ha una tendència decreixent. Això indica que, tot i les oscil·lacions pròpies d’aquest tipus de mètrica, el model està aprenent i millorant el seu rendiment amb el temps.

***


Davant l’augment progressiu de la pèrdua de validació observat en l’execució anterior, indicatiu d’overfitting, plantegem una nova execució aplicant tècniques de regularització per millorar la capacitat de generalització del model. En concret, ajustem alguns paràmetres clau com el dropout i el weight decay, i incorporem l’estratègia ReduceLROnPlateau per reduir la taxa d’aprenentatge quan la loss de validació deixa de millorar. L’objectiu és reduir la diferència entre la pèrdua d’entrenament i la de validació.

<img src="imatges\img_3.png" alt="img" width="500"/> 



En aquesta nova execució del model, s’han incorporat diverses millores per tal d’afavorir una millor generalització i estabilitat durant l’entrenament. Concretament, s’ha fixat el nombre d’èpoques en 30, amb un mecanisme d’early stopping de paciencia 6 per evitar un entrenament innecessàriament llarg si no hi ha millores. La learning rate inicial s’ha establert a 1e-3, amb un weight decay de 1e-4 aplicat a l’optimitzador Adam com a tècnica de regularització L2. També s’ha afegit un scheduler de tipus ReduceLROnPlateau, configurat amb patience=3 i factor=0.5, per reduir dinàmicament la taxa d’aprenentatge quan la pèrdua de validació es manté estable. A nivell d’arquitectura, el model CNN-GRU utilitza ara una dimensió oculta de 128 neurones i un dropout del 40%, per ajudar a reduir el sobreajustament.

En el primer gràfic, on es mostren les corbes de la loss d’entrenament i validació, observem una millora clara respecte a l’execució anterior. Les dues corbes decreixen de manera més paral·lela i sostinguda, i la val_loss no presenta pujades constants ni oscil·lacions marcades. Aquesta millora en l’estabilitat es pot atribuir a la combinació dels ajustos esmentats prèviament, que han afavorit un aprenentatge més eficient i una millor capacitat de generalització.

En el segon gràfic, que mostra l’evolució del learning rate, podem veure com aquest es redueix en dues fases ben diferenciades. Aquest comportament confirma l’acció del mecanisme ReduceLROnPlateau, que intervé quan la millora s’estanca. A més, després de cada reducció, s’observa una disminució corresponent en la val_loss, fet que indica que la reducció adaptativa de la taxa d’aprenentatge ha tingut un efecte positiu sobre la convergència del model.

Pel que fa als dos últims gràfics, la mètrica de val_accuracy mostra una tendència ascendent clara i sostinguda, arribant a superar el 65%, mentre que la batch_loss decreix de manera general amb menys oscil·lacions que en l’execució anterior. Aquest comportament conjunt reforça la idea que el model ha assolit una convergència més estable i eficient, gràcies a la combinació de regularització, adaptació dinàmica del learning rate i millor configuració d’hiperparàmetres.


***


En aquesta nova execució, hem mantingut totes les millores implementades anteriorment, com l’ús del *ReduceLROnPlateau*, la regularització amb *weight decay*, etc., però hem fet dues modificacions addicionals amb l’objectiu de seguir millorant la capacitat del model per generalitzar.

D’una banda, hem aplicat **data augmentation** sobre els espectrograms per incrementar la variabilitat de les mostres d’entrenament. Aquesta tècnica permet exposar el model a versions una mica alterades de les mateixes dades, ajudant-lo així a no sobreajustar-se i a ser més robust .

D’altra banda, hem realitzat una **selecció dels gèneres** musicals presents al dataset. Originalment comptàvem amb 16 gèneres, però hem eliminat aquells tracks que pertanyen als gèneres menys freqüents a l'hora de fer l'entrenament, concretament: *blues*, *easy listening*, *electronic* i *soul-RnB*. Aquesta reducció ens permet centrar l’entrenament en classes amb representació més equilibrada, millorant potencialment el rendiment del model.

<img src="imatges\img_4.png" alt="img" width="500"/> 


En el primer gràfic podem observar les corbes de train loss i validation loss mostren una disminució clara i sostinguda al llarg de l’entrenament. La pèrdua de validació decreix de forma regular, i es manté per sota de la d'entrenament durant gairebé tot el procés, cosa que indica que el model no està sobreajustant-se i que generalitza correctament. Aquesta millora pot atribuir-se tant al data augmentation, que incrementa la robustesa del model, com a la selecció de gèneres, que redueix la confusió causada per categories poc representatives o amb mostres molt limitades.

També veiem en el segon gràfic que el learning rate s'ha mantingut constant, fet que associem a que com que la val_loss ha anat disminuint de forma consistent, no hi ha hagut necessitat d'ajustar la taxa d’aprenentatge.

A més, observem una milloria en l'accuracy, augmentant a un 69%, i es manté bastant estable al voltant d’aquest valor. Aquest comportament indica que el model ha après a generalitzar bé, degut al data augmentation i a la selecció dels gèneres més consistents, reduint la complexitat i l’ambigüitat del problema de classificació.

La pèrdua per batch presenta una tendència descendent clara, molt similar al de l'execució anterior, cosa que suggereix que les millores aplicades prèviament (com la regularització, el scheduler i l'optimitzador Adam) segueixen funcionant adequadament.

***

Després d’haver obtingut bons resultats amb l’arquitectura basada en CNN1D + GRU bidireccional, vam voler provar una nova arquitectura. En concret, hem mantingut la mateixa estructura de convolucions 1D, però substituint la capa GRU bidireccional per una LSTM unidireccional.  Aquesta configuració redueix la complexitat del model i ens permet avaluar si una LSTM en una sola direcció, sense combinar estats temporals intermedis, pot ser suficient per capturar la informació necessària per a la classificació de gènere musical.

<img src="imatges\img_5.png" alt="img" width="500"/> 


Pel que fa als resultats, observem una millora progressiva tant en la pèrdua d’entrenament com en la pèrdua de validació, les quals disminueixen de manera consistent al llarg dels passos, sense senyals clares d’overfitting. El gràfic de learning_rate mostra una caiguda abrupta al final, fet que fa que la loss tant de l'entrenament com de la validació disminueixin una mica més.
La precisió de validació augmenta inicialment de forma pronunciada fins assolir valors propers al 68%, i després es manté relativament estable amb lleugeres oscil·lacions, indicant una certa estabilitat en l’aprenentatge.  Finalment, la batch_loss presenta molta variabilitat, però amb una tendència general a la baixa, indicant que el model continua aprenent dins de cada època. 


Ara vam voler provar una nova arquitectura utilitzant LSTM bidireccional, amb l’objectiu de captar millor les dependències temporals en totes dues direccions. A diferència de la prova anterior, després de l’LSTM apliquem dues capes de pooling global, un average pooling i un max pooling, i concatenem els resultats abans de passar-los a les capes finals de classificació. A més, hem ajustat el comportament de l’aprenentatge amb un ReduceLROnPlateau menys permissiu, utilitzant un factor de 0.7, un patience de 2 i un llindar de millora relatiu de 1e-3, per tal de reduir el learning rate de manera més ràpida quan la lossde validació s’estanca. Amb aquesta configuració, esperem una adaptació més àgil i una millor convergència del model.

<img src="imatges\img_6.png" alt="img" width="500"/> 

Podem observar que tant la corba de loss d'entrenament com la de validació continuen disminuint de manera progressiva al llarg de l'entrenament. A més, la loss de validació es manté per sota de la d'entrenament, cosa que indica que el model no pateix overfitting, sinó que seguim generalitzant bé. Pel que fa al learning rate, es produeixen dues baixades durant l'entrenament, les quals afavoreixen una reducció suau però més pronunciada de les pèrdues en els trams posteriors. En relació amb la accuracy de validació, observem un comportament molt similar al de l'execució anterior: un augment ràpid a l'inici seguit d'una estabilització al voltant del 68%, la qual cosa mostra una consistència en el rendiment del model. Finalment, la batch loss també decreix de forma general, tot i que presenta força variabilitat amb pics.

Ara realitzem una nova execució provant amb una reducció del learning rate menys permissiva, ajustant el paràmetre threshold a 1e-4 i establint threshold_mode='abs'. Amb aquests canvis, el criteri per a que el learning rate disminueixi es torna més estricte. No obstant això, com que observem que a partir de la època 10 la millora es comença a estancar, considerem aplicar una estratègia alternativa com StepLR amb un pas de 10 èpoques, que redueix el learning rate de forma més sistemàtica i controlada. Aquesta combinació ens pot ajudar a seguir reduint la loss i millorar lleugerament la accuracy.

<img src="imatges\img_7.png" alt="img" width="500"/> 

Com en els gràfics que s'han vist anteriorment, seguim amb una evolució positiva del model, amb una disminució clara tant de la loss d’entrenament com de validació, que es mantenen pròximes i sense indicis d’overfitting. Aquesta millora es pot atribuir al fet que les losses no s’estanquen tant com abans gràcies a l’ajust del StepLR, que ara redueix el learning rate aproximadament cada 10 èpoques, permetent una millor optimització. Aquest ajust també ha contribuït a l’increment de la val_accuracy, que ha arribat fins al 69%, indicant una millora notable en la capacitat de generalització del model.


***

Davant dels bons resultats obtinguts amb l’ús del StepLR, decidim fer un grid search per optimitzar encara més el rendiment d’aquesta xarxa i explorar el seu potencial. Aquesta cerca ens ha permès ajustar diversos hiperparàmetres per trobar la combinació més eficaç:

- **Model**: CNN1D-LSTM 
- **batch_size**: 32 
- **dropout**: 0.4
- **hidden_dim**: 256  
- **learning_rate**: 0.0005 
- **num_layers**: 2  
- **reduce_lr_factor**: 0.7
- **step_lr_gamma**:0.7
- **step_lr_size**:10
- **Dataset**: FMA_large (`./fma_mel_features.h5`)  

Aquests han sigut els resultats de l'execució del model amb els valors dels hiperparàmetres obtinguts mitjançant el grid search:

<img src="imatges\img_8.png" alt="img" width="500"/> 

Els gràfics mostren una evolució molt positiva del model, ja que tant la training loss com la validation loss disminueixen de forma sostinguda i propera, sense senyals d’overfitting (com hem vist fins ara). La validation accuracy assoleix un valor molt destacat de 0.713, el qual és la més alta que hem obtingut fins ara, cosa que reafirma el bon comportament del stepLR i la importància de fer un grid search. Aquesta millora també es veu reflectida en una validation loss baixa de 0.934 i una test accuracy final de 0.69754. Aquestes mètriques indiquen que el model generalitza millor i treu profit dels ajustos aplicats.


