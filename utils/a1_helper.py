
def generate_rule_A1_helper_x(mean, std):
    r1 = lambda x,y: x <= mean-std
    r2 = lambda x,y: mean-std < x and x <= mean
    r3 = lambda x,y: mean < x and x <= mean+std
    r4 = lambda x,y: x > mean+std
    return [r1, r2, r3, r4]

def generate_rule_A1_helper_y(mean, std):
    r1 = lambda x,y: y <= mean-std
    r2 = lambda x,y: mean-std < y and y <= mean
    r3 = lambda x,y: mean < y and y <= mean+std
    r4 = lambda x,y: y > mean+std
    return [r1, r2, r3, r4]
