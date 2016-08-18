import numpy as np

def normc(mat):
    return np.divide(mat, np.sqrt(np.sum(np.square(mat), axis=0)))
    
def compute_light_vector(iota):
    light_vec = np.array([np.cos(iota[14])*np.sin(iota[13]), np.sin(iota[14]), np.cos(iota[14])*np.cos(iota[13])])
    return normc(light_vec)

def compute_illumination_and_color_parameters(iota):

    # Color gain and contrast
    G = np.diag([iota[0], iota[1], iota[2], 1])
    L = np.array([[0.3, 0.59, 0.11, 0],
                  [0.3, 0.59, 0.11, 0],
                  [0.3, 0.59, 0.11, 0],
                  [0,      0,    0, 1]])

    C = np.eye(4) + (1-iota[3])*L
    M = np.dot(G, C)

    # Color offset
    Mo = np.array([[1, 0, 0, iota[4]],
                  [0,  1, 0, iota[5]],
                  [0,  0, 1, iota[6]],
                  [0,  0, 0, 1]])

    # Color correction matrix
    Cc = np.dot(Mo, M)

    # Ambient light matrix
    L_amb = np.diag([iota[7], iota[8], iota[9]])

    # Directed light matrix
    L_dir = np.diag([iota[10], iota[11], iota[12]])

    # Specular light matrix and parameters
    L_spec = np.diag([1-iota[10], 1-iota[11], 1-iota[12]])
    ks = iota[15]
    v = iota[16]

    return G, L, C, M, Mo, Cc, L_amb, L_dir, L_spec, ks, v

def compute_illumination_vectors(W, N, light_vector):
   
    # Reflection formula
    R = normc(2*np.dot(light_vector, N.T)*N.T - np.tile(light_vector, (N.shape[0], 1)).T)

    # ?
    V = -normc(W)
                                   
    return R.T, V

def illuminate_points(T, N, R, V, light_vector, ks, v, L_amb, L_dir, L_spec):
    
    ambient_light = np.dot(L_amb, T.T)

    diffuse_dot= np.tile(np.sum(N*light_vector, axis = 1), (3, 1))
    diffuse_term = diffuse_dot * T.T
    diffuse_light = np.dot(L_dir, diffuse_term)

    specular_dot = np.tile(np.sum(R*V, axis=1), (3, 1))
    specular_term = ks * specular_dot**v
    specular_light = np.dot(L_spec, specular_term)

    I = ambient_light + diffuse_light + specular_light
    
    return diffuse_term, diffuse_dot, specular_term, specular_dot, I