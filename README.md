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

## TODO:
- [x] Implement Adam
- [x] MAF in train
- [x] Projected Gradient Descent (Adam)
- [ ] Use Pytorch loss function
- [x] Generate rules
- [ ] Generalize lambda functions
- [ ] Generalize model for datasets

## Comments:
Pytorch implementation working withou projecting masses. When trying to project, we have a backward error and if we fix it, the masse values barely change over iterations (and take a lot longer to process)

Returning mass (instead of prob) yields the best results so far. However, uncertainty is as high as the prob of predicted class... (falar de varias alternativas para calculo de probabilidade)

Not considering uncertainty makes the loss go down faster, when using A1 dataset... (considerar uncertanty como class)
Using dataset or two points has the same tendency

Projected masses can be assessed but do not translate into better results (alterantivas para projecao de massas)

Using article generated rules gives good results for mass and uncertainty belief
