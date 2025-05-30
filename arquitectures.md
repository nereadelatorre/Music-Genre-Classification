# Arquitectures implementades
En aquest document es presenten les diferents arquitectures que s'han utilitzat en el projecte. Aquestes combinen una xarxa convolucional (CNN) per extreure característiques del senyal d’àudio amb una xarxa recurrent (RNN) per modelar la seva evolució temporal i/o freqüencial. Totes les arquitectures parteixen d’un **espectrograma Mel** d’entrada i fan una classificació final en 16 categories.
Les arquitectures implementades han estat les següents:
1. **CNN-1D + GRU** 
2. **CNN-1D + LSTM** 
3. **CNN-2D + LSTM** 
4. **CNN-2D + GRU** 

## Per què GRU i LSTM?
Vam decidir provar tant GRU (Gated Recurrent Unit) com LSTM (Long Short-Term Memory) perquè, tot i ser similars, tenen diferències importants:

- **L’LSTM** utilitza una arquitectura amb cel·les de memòria i tres portes (input, forget i output), que li permet controlar de manera molt precisa quina informació es conserva o s’oblida al llarg del temps. Aquesta capacitat fa que sigui especialment potent per captar dependències llargues dins d’una seqüència, com pot ser una progressió harmònica que apareix de manera esporàdica o petits canvis en la estructura musical. 
  Com volíem capturar patrons temporals més llargs i complexos dins dels espectrograms Mel, que poden ser claus per diferenciar gèneres musicals amb estructures més riques o canviants, com el jazz o la música clàssica. 
- **El GRU** és una alternativa més simple a l’LSTM. Utilitza només dues portes (reset i update) i no disposa d’una cel·la de memòria separada. Aquesta simplicitat fa que el GRU sigui més lleuger i més ràpid d’entrenar, amb menys paràmetres, però igualment capaç de capturar dependències temporals en moltes tasques seqüencials.
  En el nostre cas, volíem provar una arquitectura més eficient i que pogués generalitzar bé amb menys recursos computacionals. Això és especialment útil en problemes com el nostre, on hi ha una gran quantitat de dades però no sempre és necessari capturar contextos molt llargs. 


## Arquitectures amb CNN-1D

<img src="/imatges/arquitectures/esquemaCNN1D-GRU.drawio.png" alt="Esquema GRU" width="800"/>

Tant el model amb GRU com el model amb LSTM comencen amb el mateix bloc de **convolucions 1D**, aplicades sobre l’espectrograma Mel. Aquestes convolucions analitzen el senyal durant  l'evolució del so al **llarg del temps** i tenen com a objectiu detectar patrons locals com canvis sobtats o variacions harmòniques.

El nostre extractor de característiques (feature_extractor) utilitza tres capes de convolució 1D, cada una amb més canals que l’anterior. La primera transforma els espectres originals en una representació amb hidden_dim canals. La segona i tercera augmenten aquesta capacitat a hidden_dim * 2 i hidden_dim * 4, permetent capturar patrons cada cop més abstractes i jeràrquics.

Aquesta profunditat de tres capes ens permet trobar un bon equilibri entre capacitat i eficiència computacional, evitant fer el model massa gran o lent d’entrenar.

Al acabar en cada una de les dues primeres convolucions s'aplica **MaxPool1D**. Aquesta operació redueix la llargada temporal de la seqüència a la meitat, ajudant a reduir la mida del tensor i, per tant, el cost computacional de les capes següents. A part també fem que el model sigui invariant a petits desplaçaments temporals, ja que es queda amb la resposta màxima d’un interval.

Després de cada convolució s’aplica una **normalització batch** (BatchNorm1D) per estabilitzar l’entrenament i una activació **ReLU** per afegir no-linearitat. Aquesta combinació ajuda a accelerar la convergència i evitar problemes com el gradient vanishing.

Finalment, en l’última capa convolucional s’aplica **Dropout**, ja que redueix l’overfitting al desactivar aleatòriament neurones durant l’entrenament i força el model a no dependre d’un conjunt petit de característiques.

### CNN-1D + GRU

<img src="/imatges/arquitectures/arquitecturaCNN1D-GRU.drawio.png" alt="Arquitectura GRU" width="300"/> 
<br>   


Un cop extretes les característiques amb el bloc convolucional, la seqüència resultant (amb menys passos temporals però més profunditat) s’envia a una **GRU bidireccional**.

Inicialment vam provar una GRU unidireccional, però això només tenia en compte la informació del passat. Com que la música té estructures que sovint depenen de context futur (per exemple, el final d’una frase musical pot donar sentit a l’inici), vam optar per una GRU bidireccional.

Un cop processada la seqüència amb la GRU, obtenim una representació per a cada pas temporal. En lloc d’agafar només la darrera (o concatenar la primera i l’última), vam aplicar un **mean pooling** (fem la mitjana de totes les sortides al llarg del temps), així capturem una visió global del track sencer.

Finalment, les característiques resumides es passen per un **MLP que consta de dues capes lineals** separades per una **ReLU** i un altre **Dropout**, la qual ens serveix per classificar. Encara que sigui senzilla és suficient per transformar la representació obtinguda en una predicció sobre els 16 gèneres musicals.

### CNN-1D + LSTM

<!-- ![Arquitectura LSTM](arquitecturaCNN1D-lstm.drawio.png) -->
<img src="/imatges/arquitectures/arquitecturaCNN1D-lstm.drawio.png" alt="Arquitectura LSTM" width="300"/> 


Aquesta arquitectura segueix la mateixa estratègia de processament inicial amb tres capes convolucionals com en el model amb GRU, però en lloc d’utilitzar una xarxa GRU, incorpora una **LSTM bidireccional** com a bloc recurrent. 

La sortida del bloc convolucional és una seqüència temporal de vectors, que es transforma amb permute per ajustar-se al format d’entrada de la LSTM: **[batch_size, time, features]**.

A diferència del model anterior, en aquest cas es recorre a una estratègia de dual pooling sobre les sortides de la LSTM:

- **AvgPooling**: fa la mitjana de totes les sortides al llarg del temps, capturant una representació general del contingut sonor global.

- **MaxPooling**: selecciona el valor màxim de cada característica al llarg del temps, recollint les respostes més destacades que poden indicar presència d’un gènere concret o un patró distintiu.

Aquestes dues representacions es concatenen, oferint una visió complementària i més robusta de la seqüència.

Finalment, aquestes característiques es passen per **una capa fully-connected amb ReLU i Dropout** per introduir no-linealitat i evitar sobreajustament abans de fer la classificació final.


## Arquitectures amb CNN-2D

<img src="/imatges/arquitectures/esquemaCNN2D-LSTM.drawio.png" alt="Esquema CNN2D" width="800"/> 

En lloc de tractar l’espectrograma com una simple seqüència temporal, com fem a les CNN-1D, les arquitectures amb CNN-2D consideren alhora la dimensió temporal i de freqüència. Això permet detectar patrons que poden aparèixer localment en blocs tant de temps com de freqüència, com per exemple l’inici sobtat d’un instrument, canvis de to, o textures rítmiques específiques.

D'aquesta manera, igual que abans, processem l’entrada com una imatge (temps x freqüència). Tornem a tenir 3 capes convolucionals, ja que com hem explicat en la CNN d'1 dimensió, formem una  jerarquia que permet al model anar de detalls fins a estructures globals, enriquint la representació.

Cada capa convolucional comença amb un primer **bloc convolucional 2D**. Després de cada capa, s’aplica Batch Normalization, que normalitza les activacions dins cada mini-batch, per estabilitzar l'entrenament. Seguidament trobem una ReLU, i finalment un MaxPool de mida $2\times2$ que redueix a la meitat tant la resolució temporal com la de freqüència, ampliant alhora el camp receptiu de les neurones.

Per acabar, apliquem un **Dropout2d** com hem fet a les altres dues arquitectures per millorar la capacitat de generalització i evitant l’overfitting.


### CNN-2D + GRU
<img src="/imatges/arquitectures/arquitecturaCNN2D-GRU.drawio.png" alt="Arquitectura 2D GRU" width="300"/> 

Un cop finalitzades les convolucions 2D, el tensor es reorganitza i s’aplana per formar una seqüència temporal adequada per a la GRU. Finalment, s’aplica un **mean pooling** per resumir tota la seqüència en una representació global, que es passa per un cap de classificació (Linear + ReLU + Dropout) per predir el gènere musical.

### CNN-2D + LSTM
<img src="/imatges/arquitectures/arquitecturaCNN2D-LSTM.drawio.png" alt="Arquitectura 2D LSTM" width="300"/> 

Després de les convolucions 2D, el tensor es reorganitza i s’aplana per formar una seqüència temporal on cada pas conté informació espectral important. Aquesta seqüència es processa amb una LSTM bidireccional, que captura dependències temporals llargues i contextuals dins l’àudio. 

Un cop processada la seqüència, s’apliquen operacions de **mean pooling** i **max pooling** per agregar la informació temporal en una única representació global del track. Aquesta representació es concatena i es passa pel cap de classificació per generar la predicció final.




