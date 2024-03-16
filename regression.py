import csgs
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

    X, Y, Z = generate_input_data(NPOINTS, np.random.default_rng(SEED))

    testing_table = [
            {
                'name'       : "Random superposition",
                'reference'  : lambda seed : csgs.rs(X,Y,Z,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,seed),
                'alternative': lambda seed : csgs.rs(X,Y,Z,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,seed),
                'seeds'      : [SEED]
            },
            {
                'name'       : "Gerchberg-Saxton",
                'reference'  : lambda seed : csgs.gs(X,Y,Z,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,ITERATIONS,seed),
                'alternative': lambda seed : csgs.gs(X,Y,Z,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,ITERATIONS,seed),
                'seeds'      : [SEED]
            },
            {
                'name'       : "Weighted Gerchberg-Saxton",
                'reference'  : lambda seed : csgs.wgs(X,Y,Z,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,ITERATIONS,seed),
                'alternative': lambda seed : csgs.wgs(X,Y,Z,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,ITERATIONS,seed),
                'seeds'      : [SEED]
            },
            {
                'name'       : "Compressive Sensing Gerchberg-Saxton",
                'reference'  : lambda seed : csgs.csgs(X,Y,Z,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,ITERATIONS,COMPRESSION,seed),
                'alternative': lambda seed : csgs.csgs(X,Y,Z,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,ITERATIONS,COMPRESSION,seed),
                'seeds'      : [SEED]
            },
            {
                'name'       : "Weighted Compressive Sensing Gerchberg-Saxton",
                'reference'  : lambda seed : csgs.wcsgs(X,Y,Z,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,ITERATIONS,COMPRESSION,seed),
                'alternative': lambda seed : csgs.wcsgs(X,Y,Z,FOCAL_LENGTH,PITCH,WAVELENGTH,PIXELS,ITERATIONS,COMPRESSION,seed),
                'seeds'      : [SEED]
            }
    ]

    for test in testing_table:
        for seed in test['seeds']:
            reference,   _ = test['reference'](seed)
            alternative, _ = test['alternative'](seed)

            err = np.max(np.abs(reference - alternative))
            if (err > 0.0):
                print(f"Something went south in testing {test['name']}, L-inf error {err}")

