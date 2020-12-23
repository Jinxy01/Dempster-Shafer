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
- [ ] Projected Gradient Descent (Adam)

## Comments:
Pytorch implementation working withou projecting masses. When trying to project, we have a backward error and if we fix it, the masse values barely change over iterations (and take a lot longer to process)

Added a rule to promote combined masses. It works ok