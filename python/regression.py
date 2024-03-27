import csgs as legacy
import refactor
import numpy as np


def generate_input_data(npoints: int, rng: np.random.default_rng):
    x = 100.0 * (rng.random(npoints) - 0.5)
    y = 100.0 * (rng.random(npoints) - 0.5)
    z =  10.0 * (rng.random(npoints) - 0.5)

    return x, y, z


if __name__ == "__main__":
    FOCAL_LENGTH = 20.0
    PIXELS       = 512
    PITCH        = 15.0
    WAVELENGTH   = 0.488
    NPOINTS      = 100
    ITERATIONS   = 30
    COMPRESSION  = 0.05
    SEED         = 42

    rng = np.random.default_rng(SEED)

    X, Y, Z = generate_input_data(NPOINTS, np.random.default_rng(SEED))
    SPOTS = np.array([X, Y, Z]).transpose()
    PISTS = 2.0 * np.pi * rng.random(NPOINTS)


    testing_table = [
            {
                'name'       : "Random superposition",
                'reference'  : lambda seed : legacy.rs(X,Y,Z,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,seed),
                'alternative': lambda seed : refactor.rs(SPOTS,PISTS,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS),
                'seeds'      : [SEED]
            },
            {
                'name'       : "Gerchberg-Saxton",
                'reference'  : lambda seed : legacy.gs(X,Y,Z,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,ITERATIONS,seed),
                'alternative': lambda seed : refactor.gs(SPOTS,PISTS,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,ITERATIONS),
                'seeds'      : [SEED]
            },
            {
                'name'       : "Weighted Gerchberg-Saxton",
                'reference'  : lambda seed : legacy.wgs(X,Y,Z,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,ITERATIONS,seed),
                'alternative': lambda seed : refactor.wgs(SPOTS,PISTS,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,ITERATIONS),
                'seeds'      : [SEED]
            },
            {
                'name'       : "Compressive Sensing Gerchberg-Saxton",
                'reference'  : lambda seed : legacy.csgs(X,Y,Z,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,ITERATIONS,COMPRESSION,seed),
                'alternative': lambda seed : refactor.csgs(SPOTS,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,ITERATIONS,COMPRESSION,seed),
                'seeds'      : [SEED]
            },
            {
                'name'       : "Weighted Compressive Sensing Gerchberg-Saxton",
                'reference'  : lambda seed : legacy.wcsgs(X,Y,Z,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,ITERATIONS,COMPRESSION,seed),
                'alternative': lambda seed : refactor.wcsgs(SPOTS,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,ITERATIONS,COMPRESSION,seed),
                'seeds'      : [SEED]
            }
    ]


    for test in [testing_table[0:3]]:
        for seed in test['seeds']:
            reference,   _ = test['reference'](seed)
            alternative, _ = test['alternative'](seed)

            abs_err = np.max(np.abs(reference - alternative))

            nnz = np.where(reference != 0.0)
            rel_err = np.max(np.abs((reference[nnz] - alternative[nnz]) / reference[nnz]))

            print(f"Testing {test['name']}, absolute error {abs_err}, relative error {rel_err}")
