import numpy as np

# Reference: "Adam: A Method for Stochastic Optimization"
# https://arxiv.org/pdf/1412.6980.pdf
class Adam():
    def __init__(self, alpha=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_t = 0 # First moment vector
        self.v_t = 0 # Second moment vector
        self.t = 0 # timestep
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.alpha = alpha

    def update(self, theta, g_t): # g_t is result of objective function
        self.t += 1

        # Compute
        self.m_t = self.beta1*self.m_t+(1-self.beta1)*g_t
        self.v_t = self.beta2*self.v_t+(1-self.beta2)*g_t**2

        # Correct
        m_t_corrected = self.m_t/(1-self.beta1**self.t)
        v_t_corrected = self.v_t/(1-self.beta2**self.t)

        # Update parameters
        theta = theta-self.alpha*(m_t_corrected/(np.sqrt(v_t_corrected)+self.epsilon))
        return theta