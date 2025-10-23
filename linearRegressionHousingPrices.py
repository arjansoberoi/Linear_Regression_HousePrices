import numpy as np  
import matplotlib.pyplot as plt 
# X train and Y train
X_train = np.array([1.0,2.0])
Y_train = np.array([300,500])
def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = (w*x[i]) + b
        cost = cost + (f_wb - y[i]) ** 2
    total_cost = 1/(2*m) * cost
    return total_cost
def compute_gradient(x,y,w,b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = (w*x[i]) + b 
        dj_dw_temp = (f_wb - y[i]) * x[i]
        dj_db_temp = (f_wb - y[i])
        dj_dw += dj_dw_temp 
        dj_db += dj_db_temp
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db
def gradient_descent(x,y,w_init,b_init,alpha, num_iters, cost_function,gradient_function):
    j_history = []
    p_history = []
    b = b_init
    w = w_init
    for i in range(num_iters):
        dj_dw,dj_db = gradient_function(x,y,w,b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 100000:
            j_history.append(cost_function(x,y,w,b))
            p_history.append([w,b])
        if i % 100 == 0:
            cost = cost_function(x, y, w, b)
            print(f"Iteration {i:4d}: Cost {cost:8.2f}")
    return w,b,j_history,p_history
#testing 
w_init = 0
b_init = 0
alpha = 0.01
num_iters = 10000
w_final,b_final,J_hist,P_hist = gradient_descent(X_train,Y_train,w_init,b_init,alpha,num_iters,compute_cost,compute_gradient )
print(f"The weight is: {w_final:8.4f} and the bias is: {b_final:8.4f}")
print(f"a 1200 sqft house is {w_final*1.2 + b_final:0.1f}")
print(f"a 2000 sqrt house is {w_final*2.0 + b_final:0.1f} thousand dollars")
# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
plt.show()