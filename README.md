# Dempster-Shafer

## Installation
1. Create venv and use it (Optional):
```bash
python3 -m venv venv
source venv/bin/activate
```
2. Install requirements:
```bash
pip3 install -r requirements.txt
```
3. Run bash script `test.sh`:
```bash
bash test.sh
```
Any errors with "missing Python.h", when installing sklearn, run: 
```bash
sudo apt-get install build-essential python3-dev python3-pip
```

Any errors with "UserWarning: Matplotlib is currently using agg, which is a non-GUI backend", when displaying graph, run: 
```bash
sudo apt-get install python3-tk
```

## TODO:
- [x] Implement Adam
- [x] MAF in train
- [x] Projected Gradient Descent (Adam)
- [x] Use Pytorch loss function
- [x] Generate rules
- [x] Generalize lambda functions of A1
- [ ] Generalize lambda functions of BC
- [x] Generalize model for datasets
- [ ] Add belief to inference
- [x] Update frozenset_to_class to consider dataset_name
- [ ] Check what rule of BC corresponds to
- [x] Iris dataset: use non optimzed stuff!

## Comments:
* Pytorch implementation working withou projecting masses. When trying to project, we have a backward error and if we fix it, the masse values barely change over iterations (and take a lot longer to process)
* Ways to return probabilities
* Alterantivas para projecao de massas
* Using article generated rules gives good results for mass and uncertainty belief
* Diferentes modos para retornar prob de classes (prob para todas vs prob para a mais alta = dif no graph de loss)
* Wrong (em stats e imgs) = quando a prob de classes era mal calculada (so devolvida prob da class mais provavel e nao de todas)
* Quando correr IRIS, retirar otimizacoes, em dempster_shaffer.py (para poder ser capaz de lidar com varias classes), mudar NUM_CLASSES em config.py e descomentar CLASS_2_ONE_HOT

Uso de profiler para ajudar a perceber limitacoes de implementacao
Optimizações:
* MSE (stack)
* Dempster Shaffer usando commody 

Em stats:
* A1_ds_mse_optim = dempster shaffer otimizado e mse de pytorch
* A1_pl_ds_mse_optim = igual a anterior mas com plausability melhorado (sem usar plausibility_set)
* A1_inf_pl_ds_mse_optim = igual a anterior mas com inferencia melhorada (sem usar weight_uncertainty)
* A1_ohe_inf_pl_ds_mse_optim = igual a anterior mas com one hot enconding de classes 0 e 1 em config.py

* BC_full_optim = tem todas as otimizações

    

