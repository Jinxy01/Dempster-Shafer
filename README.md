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
- [ ] MAF in train
- [ ] Projected Gradient Descent (Adam)

## Comments:
The way we are updating weights, we are only reducing every parameter (R,B and R+B). This will decrease loss since R+B (uncertanty) is also decreasing. However, if we normalize we get uncertanty to go higher again, while reducing R and B...

Added pytorch implementation in testing_stuff_t(). Article says to compute **belief** but that way we only update R and B mass values... Made some changes and now R_B is also changed but still not good enough