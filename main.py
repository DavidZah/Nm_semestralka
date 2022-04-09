import numpy as np
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

const = 50


def momentum_gradient_descend(start,n_steps=2000,step_size = 0.001,epsilon = 0.001,momentum = 0.9):
    epsilon=epsilon*step_size
    path = []
    old_vals = []
    pos = start


    change = 0
    for i in range(0,n_steps):
        val = fcn_non_lin(pos)
        old_vals.append(val)
        old_change=change
        change = step_size*np.array(fcn_non_lin_grad(pos))+momentum*old_change
        old_pos = pos
        pos = pos-change

        path.append(pos)
        if(np.allclose(old_pos,pos,rtol=epsilon)):
            break
    print(pos)
    plot_2D_grad_desced_non_lin(path)



def fcn_non_lin(x):
    """
    lin_part = x[0]*x[0]+x[1]*x[1]
    sin_part = const*np.sin(x[0])+const*np.cos(x[1])
    ret = lin_part+sin_part+const
    """
    ret = x[0]*x[0]+x[1]*x[1]
    return ret

def fcn_non_lin_grad(x):
    """
    ret = np.array([2*x[0]+const*np.cos(x[0]),-2*x[1]*const*np.sin(x[1])])
    ret = ret*-1
    """
    ret = [2*x[0],2*x[1]]

    return ret

def simple_gradient_desend_non_lin(start,n_steps=1000,step_size = 0.001,epsilon = 0.001):
    epsilon=epsilon*step_size
    path = []
    old_vals = []
    pos = start

    for i in range(0,n_steps):
        val = fcn_non_lin(pos)
        old_vals.append(val)
        direction = step_size*fcn_non_lin_grad(pos)
        old_pos = pos
        pos = pos + direction
        path.append(pos)
        if(np.allclose(old_pos,pos,rtol=epsilon)):
            break
    print(pos)
    plot_2D_grad_desced_non_lin(path)

def plot_2D_grad_desced_non_lin(path,size=[-20,20],step = 0.1):

    mat = []
    X = []
    Y = []
    for i in np.arange (size[0],size[1],step):
            oper_arr = []
            for j in np.arange(size[0], size[1], step):

                oper_arr.append(fcn_non_lin(np.array([i,j])))
            mat.append(oper_arr)

    mat= np.array(mat)

    x_path = []
    y_path = []
    z_path = []

    for i in path:
        x_path.append(i[0])
        y_path.append(i[1])
        z_path.append(fcn_non_lin(np.array(i)))

    x = np.outer(np.linspace(size[0],size[1],mat.shape[0]), np.ones(mat.shape[0]))
    y = x.copy().T

    fig = plt.figure(figsize=(abs(size[0]), abs(size[1])))
    ax = plt.axes(projection='3d')

    """
    plt.contour(x, y, mat, 20,cmap='RdGy');
    plt.plot(x_path,y_path)
    plt.colorbar();
    plt.show()
    """
    ax.plot(x_path, y_path, z_path, color='red', linewidth=1)
    ax.plot_surface(x,y, mat,alpha = 0.7)
    plt.show()



def fcn_quat_grad(A,b,x):
    A_part = 0.5*(np.linalg.multi_dot([np.transpose(A),x]))
    B_part = 0.5*(np.linalg.multi_dot([A,x]))
    C_part = np.add(A_part,B_part)
    return np.array(-1*np.subtract(C_part,b))

def fcn_quat(A,b,x):
    A_part = np.linalg.multi_dot([np.transpose(x),A,x])
    B_part = np.linalg.multi_dot([np.transpose(b),x])
    return 0.5*np.subtract(A_part, B_part)

def plot_2D_grad_desced(path,A,b,size=[-20,20]):

    mat = []
    X = []
    Y = []
    for i in range(size[0],size[1]):
            oper_arr = []
            for j in range(size[0],size[1]):

                oper_arr.append(fcn_quat(A,b,np.array([i,j])))
            mat.append(oper_arr)

    fig = plt.figure(figsize=(abs(size[0]),abs(size[1])))
    ax = plt.axes(projection='3d')
    mat= np.array(mat)

    x_path = []
    y_path = []
    z_path = []

    for i in path:
        x_path.append(i[0])
        y_path.append(i[1])
        z_path.append(fcn_quat(A,b,np.array(i))+1)

    step = abs(size[0])+abs(size[1])

    x = np.outer(np.linspace(size[0],size[1], step), np.ones(step))
    y = x.copy().T


    ax.plot(x_path, y_path, z_path, color='red', linewidth=1)
    ax.plot_surface(x,y, mat,alpha = 0.7)
    plt.show()

def simple_gradient_desend(A,b,start,n_steps=100,step_size = 0.1,epsilon = 0.01):
    path = []
    pos = start

    for i in range(0,n_steps):
        direction = step_size*fcn_quat_grad(A,b,pos)

        old_pos = pos
        pos = pos + direction
        pos = pos[0]
        path.append(pos)
        if(np.allclose(old_pos,pos,rtol=epsilon)):
            break
    print(pos)
    plot_2D_grad_desced(path,A,b,size=[-10,10])

if __name__ == "__main__":
    A = np.matrix('1 0; 0 1')
    b = np.array([0, 0])
    x = np.array([0,0])

    #simple_gradient_desend(A,b,np.array([10,8]))

    #simple_gradient_desend_non_lin([5,5])
    momentum_gradient_descend([20,20])
    print("done")